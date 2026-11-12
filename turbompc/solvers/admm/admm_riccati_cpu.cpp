// Fused CPU ADMM with a structure-exploiting (Riccati-type) factorization.
//
// Runs the same ADMM iteration as ADMMSolver's JAX loop with projection handling
// (no slacks), for a fixed number of iterations, entirely in native code:
//
//   x-update  (P + sigma I + rho G'G + rho_f C'C) x = gamma
//   gamma     = sigma x - q + C'(rho_f c - y_f) + G'(rho z - y_g)
//   z, y_g, y_f updates with over-relaxation alpha
//
// The x-update matrix is factored once per call (rho is fixed within a call) and every
// iteration only substitutes. The factorization is a block Cholesky ordered like a
// Riccati recursion: when the dynamics couple consecutive nodes only through the next
// state (A_plus = [A_next, 0], no cost cross terms E), the controls of every node are
// eliminated first, independently of each other, and what remains is a block-tridiagonal
// chain over the states alone (nx instead of nx + nu per node). Any other structure
// falls back to the same chain over whole nodes.
//
// Node-independent work (assembly, control elimination, the per-node parts of every
// iteration) is spread over a small persistent thread pool; the state chain is sequential.

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <xla/ffi/api/c_api.h>
#include <xla/ffi/api/ffi.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

namespace ffi = xla::ffi;

namespace {

using Index = Eigen::Index;

// [primal residual, dual residual, primal norm term, dual norm term, finite, factored,
//  structure (0 whole nodes, 1 controls first, 2 controls first with A_plus = [-I 0]),
//  threads, factorization seconds, iteration seconds]
constexpr Index kInfoSize = 10;

inline void CpuRelax() {
#if defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#endif
}

// ----------------------------------------------------------------------------------
// Persistent fork-join pool. The calling thread runs chunk 0; workers run the others.
// Workers spin for a short while after each task (phases within one solve follow each
// other within microseconds) and sleep on a condition variable otherwise.
class ThreadPool {
 public:
  static ThreadPool& Instance() {
    static ThreadPool* pool = new ThreadPool();  // never destroyed: workers outlive statics
    return *pool;
  }

  // Calls fn(chunk, begin, end) for `chunks` contiguous slices of [0, count).
  template <typename Fn>
  void Run(Index count, int chunks, Fn&& fn) {
    chunks = static_cast<int>(std::max<Index>(1, std::min<Index>(chunks, count)));
    if (chunks <= 1) {
      fn(0, Index(0), count);
      return;
    }
    std::lock_guard<std::mutex> run_lock(run_mutex_);
    EnsureWorkers(chunks - 1);
    auto bounds = [count, chunks](int k) { return (count * k) / chunks; };
    Task task;
    task.call = [](void* closure, int k, Index b, Index e) {
      (*static_cast<std::remove_reference_t<Fn>*>(closure))(k, b, e);
    };
    task.closure = static_cast<void*>(&fn);
    task.count = count;
    task.chunks = chunks;
    task_ = task;
    pending_.store(static_cast<int>(workers_.size()), std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(sleep_mutex_);
      generation_.fetch_add(1, std::memory_order_release);
    }
    if (sleepers_.load(std::memory_order_acquire) > 0) {
      sleep_cv_.notify_all();
    }
    fn(0, bounds(0), bounds(1));
    while (pending_.load(std::memory_order_acquire) != 0) {
      CpuRelax();
    }
  }

 private:
  struct Task {
    void (*call)(void*, int, Index, Index) = nullptr;
    void* closure = nullptr;
    Index count = 0;
    int chunks = 1;
  };

  void EnsureWorkers(int needed) {
    while (static_cast<int>(workers_.size()) < needed) {
      const int id = static_cast<int>(workers_.size());
      // The generation before this Run's task is published: a worker that starts late still
      // sees the task as new.
      const std::uint64_t seen = generation_.load(std::memory_order_acquire);
      workers_.emplace_back([this, id, seen] { WorkerLoop(id, seen); });
      workers_.back().detach();
    }
  }

  void WorkerLoop(int id, std::uint64_t seen) {
    for (;;) {
      // Spin up to ~2 ms for the next phase, then sleep.
      auto start = std::chrono::steady_clock::now();
      int spins = 0;
      std::uint64_t now = generation_.load(std::memory_order_acquire);
      while (now == seen) {
        CpuRelax();
        if (++spins == 256) {
          spins = 0;
          if (std::chrono::steady_clock::now() - start > std::chrono::microseconds(2000)) {
            std::unique_lock<std::mutex> lock(sleep_mutex_);
            sleepers_.fetch_add(1, std::memory_order_acq_rel);
            sleep_cv_.wait(lock, [&] { return generation_.load(std::memory_order_acquire) != seen; });
            sleepers_.fetch_sub(1, std::memory_order_acq_rel);
          }
        }
        now = generation_.load(std::memory_order_acquire);
      }
      seen = now;
      const Task task = task_;
      const int k = id + 1;
      if (k < task.chunks) {
        const Index b = (task.count * k) / task.chunks;
        const Index e = (task.count * (k + 1)) / task.chunks;
        task.call(task.closure, k, b, e);
      }
      pending_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  std::mutex run_mutex_;
  std::mutex sleep_mutex_;
  std::condition_variable sleep_cv_;
  std::atomic<std::uint64_t> generation_{0};
  std::atomic<int> pending_{0};
  std::atomic<int> sleepers_{0};
  Task task_;
  std::vector<std::thread> workers_;
};

// ----------------------------------------------------------------------------------
template <typename S>
using RowMat = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <typename S>
using Mat = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;
template <typename S>
using Vec = Eigen::Matrix<S, Eigen::Dynamic, 1>;
template <typename S>
using CMapR = Eigen::Map<const RowMat<S>>;
template <typename S>
using CMapV = Eigen::Map<const Vec<S>>;
template <typename S>
using MapV = Eigen::Map<Vec<S>>;

// QP data, row-major as JAX hands it over. T nodes, n = nx + nu per node.
template <typename S>
struct Problem {
  Index T, n, nx, n0, m;
  const S *D, *E, *q, *A0, *Am, *Ap, *c0, *c, *G, *l, *u;

  CMapR<S> Dm(Index t) const { return CMapR<S>(D + t * n * n, n, n); }
  CMapR<S> Em(Index t) const { return CMapR<S>(E + t * n * n, n, n); }
  CMapV<S> qv(Index t) const { return CMapV<S>(q + t * n, n); }
  CMapR<S> A0m() const { return CMapR<S>(A0, n0, n); }
  CMapR<S> Amm(Index t) const { return CMapR<S>(Am + t * nx * n, nx, n); }
  CMapR<S> Apm(Index t) const { return CMapR<S>(Ap + t * nx * n, nx, n); }
  CMapV<S> c0v() const { return CMapV<S>(c0, n0); }
  CMapV<S> cv(Index t) const { return CMapV<S>(c + t * nx, nx); }
  CMapR<S> Gm(Index t) const { return CMapR<S>(G + t * m * n, m, n); }
  CMapV<S> lv(Index t) const { return CMapV<S>(l + t * m, m); }
  CMapV<S> uv(Index t) const { return CMapV<S>(u + t * m, m); }
};

// ADMM iterate, stored in the output buffers and updated in place.
template <typename S>
struct Iterate {
  S *x, *yg, *yf0, *yfd, *zg;
  Index T, n, nx, n0, m;
  MapV<S> xv(Index t) const { return MapV<S>(x + t * n, n); }
  MapV<S> ygv(Index t) const { return MapV<S>(yg + t * m, m); }
  MapV<S> zgv(Index t) const { return MapV<S>(zg + t * m, m); }
  MapV<S> yf0v() const { return MapV<S>(yf0, n0); }
  MapV<S> yfdv(Index t) const { return MapV<S>(yfd + t * nx, nx); }
};

template <typename S>
class Workspace {
 public:
  Workspace(Index T, Index n, Index nx, Index n0, Index m) : T_(T), n_(n), nx_(nx), n0_(n0), m_(m) {
    Dt_.assign(T, Mat<S>(n, n));
    xh_.assign(T, Vec<S>(n));
    gam_.assign(T, Vec<S>(n));
    gv_.assign(T, Vec<S>(m));
    cx_.assign(T, Vec<S>(nx));
    cx0_.resize(n0);
    Luu_.resize(T);
    W_.resize(T);
    Sd_.resize(T);
    Pt_.resize(T);
    So_.resize(T > 1 ? T - 1 : 0);
    ready_.reset(new std::atomic<std::uint64_t>[T]);
    ready_back_.reset(new std::atomic<std::uint64_t>[T]);
    for (Index t = 0; t < T; ++t) {
      ready_[t].store(0);
      ready_back_[t].store(0);
    }
    yu_.resize(T);
    a_.resize(T);
    b_.resize(T);
    v_.resize(T);
    s_.resize(T);
    llt_uu_.resize(T);
    llt_ss_.resize(T);
  }

  bool matches(Index T, Index n, Index nx, Index n0, Index m) const {
    return T == T_ && n == n_ && nx == nx_ && n0 == n0_ && m == m_;
  }

  std::mutex mutex;

  // Returns false if a pivot block is not positive definite. With threads > 1 the per-node work
  // runs on the pool's workers, in node order, while this thread runs the state chain behind them.
  bool Factorize(const Problem<S>& P, S sigma, S rho, S rho_f, int threads) {
    const Index T = T_, n = n_, nx = nx_;
    // Controls are eliminated first when nothing couples them across nodes.
    structured_ = nx < n;
    for (Index t = 0; structured_ && t + 1 < T; ++t) {
      if (!P.Em(t).isZero(0) || !P.Apm(t).rightCols(n - nx).isZero(0)) {
        structured_ = false;
      }
    }
    // A_plus = [-I, 0] (explicit integrators): its products are negations.
    negative_identity_ = structured_;
    for (Index t = 0; negative_identity_ && t + 1 < T; ++t) {
      if (!(-P.Apm(t).leftCols(nx)).isIdentity(0)) {
        negative_identity_ = false;
      }
    }
    ms_ = structured_ ? nx : n;
    mu_ = n - ms_;
    const Index ms = ms_, mu = mu_;
    for (Index t = 0; t < T; ++t) {
      if (Luu_[t].rows() != mu) {
        Luu_[t].resize(mu, mu);
        W_[t].resize(2 * ms, mu);
        yu_[t].resize(mu);
      }
      if (Sd_[t].rows() != ms) {
        Sd_[t].resize(ms, ms);
        Pt_[t].resize(ms, ms);
        a_[t].resize(ms);
        b_[t].resize(ms);
        v_[t].resize(ms);
        s_[t].resize(ms);
      }
      if (t + 1 < T && So_[t].rows() != ms) {
        So_[t].resize(ms, ms);
      }
    }
    if (threads <= 1) {
      for (Index t = 0; t < T; ++t) {
        if (!NodeFactor(P, sigma, rho, rho_f, t)) {
          return false;
        }
      }
      for (Index t = 0; t < T; ++t) {
        if (!ChainFactor(t)) {
          return false;
        }
      }
      return true;
    }
    const std::uint64_t epoch = ++epoch_;
    next_.store(0, std::memory_order_relaxed);
    std::atomic<bool> ok{true};
    ThreadPool::Instance().Run(threads, threads, [&](int k, Index, Index) {
      if (k == 0) {
        for (Index t = 0; t < T; ++t) {
          while (ready_[t].load(std::memory_order_acquire) != epoch) {
            CpuRelax();
          }
          if (ok.load(std::memory_order_relaxed) && !ChainFactor(t)) {
            ok.store(false, std::memory_order_relaxed);
          }
        }
        return;
      }
      for (Index t; (t = next_.fetch_add(1, std::memory_order_relaxed)) < T;) {
        if (!NodeFactor(P, sigma, rho, rho_f, t)) {
          ok.store(false, std::memory_order_relaxed);
        }
        ready_[t].store(epoch, std::memory_order_release);
      }
    });
    return ok.load();
  }

  // Per node: assemble the diagonal block (lower triangle), eliminate the node's controls, and
  // the node's terms of the state chain: S_tt (without the previous node's), S_{t+1,t}, and
  // P_t = W1 W1' that the next node's S_tt subtracts.
  bool NodeFactor(const Problem<S>& P, S sigma, S rho, S rho_f, Index t) {
    const Index T = T_, nx = nx_, ms = ms_, mu = mu_;
    Mat<S>& Dt = Dt_[t];
    Dt = P.Dm(t);
    Dt.diagonal().array() += sigma;
    if (m_ > 0) {
      Dt.template selfadjointView<Eigen::Lower>().rankUpdate(P.Gm(t).transpose(), rho);
    }
    if (t + 1 < T) {
      Dt.template selfadjointView<Eigen::Lower>().rankUpdate(P.Amm(t).transpose(), rho_f);
    }
    if (t > 0) {
      if (negative_identity_) {
        Dt.topLeftCorner(nx, nx).diagonal().array() += rho_f;
      } else if (structured_) {
        Dt.topLeftCorner(nx, nx).template selfadjointView<Eigen::Lower>().rankUpdate(
            P.Apm(t - 1).leftCols(nx).transpose(), rho_f);
      } else {
        Dt.template selfadjointView<Eigen::Lower>().rankUpdate(P.Apm(t - 1).transpose(), rho_f);
      }
    }
    if (t == 0 && n0_ > 0) {
      Dt.template selfadjointView<Eigen::Lower>().rankUpdate(P.A0m().transpose(), rho_f);
    }
    Sd_[t] = Dt.topLeftCorner(ms, ms);
    if (mu > 0) {
      llt_uu_[t].compute(Dt.bottomRightCorner(mu, mu));
      if (llt_uu_[t].info() != Eigen::Success) {
        return false;
      }
      Luu_[t] = llt_uu_[t].matrixL();
      // W = [H_su; E_su] L_uu^{-T}: the controls' coupling to this and the next state.
      Mat<S>& W = W_[t];
      W.topRows(ms) = Dt.bottomLeftCorner(mu, ms).transpose();
      if (t + 1 < T && negative_identity_) {
        W.bottomRows(ms) = -rho_f * P.Amm(t).rightCols(mu);
      } else if (t + 1 < T) {
        W.bottomRows(ms).noalias() = rho_f * (P.Apm(t).leftCols(nx).transpose() * P.Amm(t).rightCols(mu));
      } else {
        W.bottomRows(ms).setZero();
      }
      Luu_[t].transpose().template triangularView<Eigen::Upper>().template solveInPlace<Eigen::OnTheRight>(W);
      Sd_[t].template selfadjointView<Eigen::Lower>().rankUpdate(W.topRows(ms), S(-1));
      if (t + 1 < T) {
        Pt_[t].setZero();
        Pt_[t].template selfadjointView<Eigen::Lower>().rankUpdate(W.bottomRows(ms), S(1));
      }
    }
    if (t + 1 < T) {
      if (negative_identity_) {
        So_[t] = -rho_f * P.Amm(t).leftCols(nx);
        So_[t].noalias() -= W_[t].bottomRows(ms) * W_[t].topRows(ms).transpose();
      } else if (structured_) {
        So_[t].noalias() = rho_f * (P.Apm(t).leftCols(nx).transpose() * P.Amm(t).leftCols(nx));
        So_[t].noalias() -= W_[t].bottomRows(ms) * W_[t].topRows(ms).transpose();
      } else {
        So_[t] = P.Em(t);
        So_[t].noalias() += rho_f * (P.Apm(t).transpose() * P.Amm(t));
      }
    }
    return true;
  }

  // Chain step t (sequential): the previous node's terms, L_tt, and M_t = S_{t+1,t} L_tt^{-T}.
  bool ChainFactor(Index t) {
    if (t > 0) {
      if (mu_ > 0) {
        Sd_[t].template triangularView<Eigen::Lower>() -= Pt_[t - 1];
      }
      Sd_[t].template selfadjointView<Eigen::Lower>().rankUpdate(So_[t - 1], S(-1));
    }
    llt_ss_[t].compute(Sd_[t]);
    if (llt_ss_[t].info() != Eigen::Success) {
      return false;
    }
    Sd_[t] = llt_ss_[t].matrixL();  // Sd now holds L_tt (lower)
    if (t + 1 < T_) {
      Sd_[t].transpose().template triangularView<Eigen::Upper>().template solveInPlace<Eigen::OnTheRight>(So_[t]);
    }
    return true;
  }

  // One ADMM iteration. With threads > 1, as a pipeline: the workers compute gamma and the
  // control half of the forward substitution node by node while this thread runs the state chain
  // forward behind them, then back; the workers back-substitute the controls and update each node
  // as soon as the chain has passed it.
  void Iteration(const Problem<S>& P, const Iterate<S>& X, S sigma, S rho, S rho_f, S alpha, int threads) {
    const Index T = T_;
    if (threads <= 1) {
      GammaAndForward(P, X, sigma, rho, rho_f, 0, T);
      for (Index t = 0; t < T; ++t) {
        ChainForward(t);
      }
      for (Index t = T - 1; t >= 0; --t) {
        ChainBackward(t);
      }
      Backward(0, T);
      Update(P, X, rho, rho_f, alpha, 0, T);
      return;
    }
    const std::uint64_t epoch = ++epoch_;
    next_.store(0, std::memory_order_relaxed);
    next_back_.store(T - 1, std::memory_order_relaxed);
    chain_back_.store(T, std::memory_order_relaxed);
    auto back_and_update = [&]() {
      for (Index t; (t = next_back_.fetch_sub(1, std::memory_order_relaxed)) >= 0;) {
        while (chain_back_.load(std::memory_order_acquire) > t) {  // s_t and s_{t+1} known
          CpuRelax();
        }
        Backward(t, t + 1);
        ready_back_[t].store(epoch, std::memory_order_release);
        if (t + 1 < T) {
          while (ready_back_[t + 1].load(std::memory_order_acquire) != epoch) {
            CpuRelax();
          }
        }
        Update(P, X, rho, rho_f, alpha, t, t + 1);
      }
    };
    ThreadPool::Instance().Run(threads, threads, [&](int k, Index, Index) {
      if (k == 0) {
        for (Index t = 0; t < T; ++t) {
          while (ready_[t].load(std::memory_order_acquire) != epoch) {
            CpuRelax();
          }
          ChainForward(t);
        }
        for (Index t = T - 1; t >= 0; --t) {
          ChainBackward(t);
          chain_back_.store(t, std::memory_order_release);
        }
      } else {
        for (Index t; (t = next_.fetch_add(1, std::memory_order_relaxed)) < T;) {
          GammaAndForward(P, X, sigma, rho, rho_f, t, t + 1);
          ready_[t].store(epoch, std::memory_order_release);
        }
      }
      back_and_update();
    });
  }

  // gamma = sigma x - q + C'(rho_f c - y_f) + G'(rho z - y_g) for nodes [begin, end),
  // then the control-elimination half of the substitution.
  void GammaAndForward(const Problem<S>& P, const Iterate<S>& X, S sigma, S rho, S rho_f, Index begin, Index end) {
    const Index T = T_, ms = ms_, mu = mu_;
    for (Index t = begin; t < end; ++t) {
      Vec<S>& g = gam_[t];
      g.noalias() = sigma * X.xv(t) - P.qv(t);
      if (t == 0 && n0_ > 0) {
        g.noalias() += P.A0m().transpose() * (rho_f * P.c0v() - X.yf0v());
      }
      if (t + 1 < T) {
        g.noalias() += P.Amm(t).transpose() * (rho_f * P.cv(t) - X.yfdv(t));
      }
      if (t > 0) {
        if (negative_identity_) {
          g.head(nx_) -= rho_f * P.cv(t - 1) - X.yfdv(t - 1);
        } else if (structured_) {
          g.head(nx_).noalias() += P.Apm(t - 1).leftCols(nx_).transpose() * (rho_f * P.cv(t - 1) - X.yfdv(t - 1));
        } else {
          g.noalias() += P.Apm(t - 1).transpose() * (rho_f * P.cv(t - 1) - X.yfdv(t - 1));
        }
      }
      if (m_ > 0) {
        g.noalias() += P.Gm(t).transpose() * (rho * X.zgv(t) - X.ygv(t));
      }
      if (mu > 0) {
        yu_[t] = g.tail(mu);
        Luu_[t].template triangularView<Eigen::Lower>().solveInPlace(yu_[t]);
        a_[t].noalias() = W_[t].topRows(ms) * yu_[t];
        if (t + 1 < T) {
          b_[t].noalias() = W_[t].bottomRows(ms) * yu_[t];
        }
      }
    }
  }

  // Forward substitution along the state chain, node t (after node t - 1).
  void ChainForward(Index t) {
    Vec<S>& v = v_[t];
    v = gam_[t].head(ms_);
    if (mu_ > 0) {
      v -= a_[t];
      if (t > 0) {
        v -= b_[t - 1];
      }
    }
    if (t > 0) {
      v.noalias() -= So_[t - 1] * v_[t - 1];
    }
    Sd_[t].template triangularView<Eigen::Lower>().solveInPlace(v);
  }

  // Backward substitution along the state chain, node t (after node t + 1).
  void ChainBackward(Index t) {
    Vec<S>& s = s_[t];
    s = v_[t];
    if (t + 1 < T_) {
      s.noalias() -= So_[t].transpose() * s_[t + 1];
    }
    Sd_[t].transpose().template triangularView<Eigen::Upper>().solveInPlace(s);
  }

  // Back-substitute the controls of nodes [begin, end) into xh.
  void Backward(Index begin, Index end) {
    const Index T = T_, ms = ms_, mu = mu_;
    for (Index t = begin; t < end; ++t) {
      Vec<S>& xh = xh_[t];
      xh.head(ms) = s_[t];
      if (mu > 0) {
        Vec<S>& w = yu_[t];
        w.noalias() -= W_[t].topRows(ms).transpose() * s_[t];
        if (t + 1 < T) {
          w.noalias() -= W_[t].bottomRows(ms).transpose() * s_[t + 1];
        }
        Luu_[t].transpose().template triangularView<Eigen::Upper>().solveInPlace(w);
        xh.tail(mu) = w;
      }
    }
  }

  // Constraint values of the new solution, relaxation, projection and dual updates.
  void Update(const Problem<S>& P, const Iterate<S>& X, S rho, S rho_f, S alpha, Index begin, Index end) {
    const Index T = T_;
    const S one_minus = S(1) - alpha;
    for (Index t = begin; t < end; ++t) {
      const Vec<S>& xh = xh_[t];
      if (t == 0 && n0_ > 0) {
        cx0_.noalias() = P.A0m() * xh;
        X.yf0v().noalias() += rho_f * (alpha * cx0_ + one_minus * P.c0v() - P.c0v());
      }
      if (t + 1 < T) {
        Vec<S>& Cx = cx_[t];
        Cx.noalias() = P.Amm(t) * xh;
        if (negative_identity_) {
          Cx -= xh_[t + 1].head(nx_);
        } else if (structured_) {
          Cx.noalias() += P.Apm(t).leftCols(nx_) * xh_[t + 1].head(nx_);
        } else {
          Cx.noalias() += P.Apm(t) * xh_[t + 1];
        }
        X.yfdv(t).noalias() += rho_f * (alpha * Cx + one_minus * P.cv(t) - P.cv(t));
      }
      if (m_ > 0) {
        Vec<S>& gv = gv_[t];
        gv.noalias() = P.Gm(t) * xh;
        MapV<S> z = X.zgv(t);
        MapV<S> yg = X.ygv(t);
        for (Index i = 0; i < m_; ++i) {
          const S relaxed = alpha * gv[i] + one_minus * z[i];
          const S z_tilde = relaxed + yg[i] / rho;
          const S z_new = std::min(std::max(z_tilde, P.l[t * m_ + i]), P.u[t * m_ + i]);
          yg[i] += rho * (relaxed - z_new);
          z[i] = z_new;
        }
      }
      MapV<S> x = X.xv(t);
      x = alpha * xh + one_minus * x;
    }
  }

  bool structured() const { return structured_; }
  bool negative_identity() const { return negative_identity_; }

 private:
  Index T_, n_, nx_, n0_, m_;
  bool structured_ = false;
  bool negative_identity_ = false;
  Index ms_ = 0, mu_ = 0;
  std::vector<Mat<S>> Dt_;
  std::vector<Mat<S>> Luu_;
  std::vector<Mat<S>> W_;
  std::vector<Mat<S>> Sd_;
  std::vector<Mat<S>> Pt_;
  std::vector<Mat<S>> So_;
  // Pipeline bookkeeping: per-node ready marks hold the epoch of the phase that produced them.
  std::uint64_t epoch_ = 0;
  std::unique_ptr<std::atomic<std::uint64_t>[]> ready_, ready_back_;
  std::atomic<Index> next_{0}, next_back_{0}, chain_back_{0};
  std::vector<Eigen::LLT<Mat<S>, Eigen::Lower>> llt_uu_;
  std::vector<Eigen::LLT<Mat<S>, Eigen::Lower>> llt_ss_;
  std::vector<Vec<S>> xh_, gam_, gv_, cx_, yu_, a_, b_, v_, s_;
  Vec<S> cx0_;
};

template <typename S>
Workspace<S>& FindWorkspace(Index T, Index n, Index nx, Index n0, Index m) {
  static std::mutex registry_mutex;
  static std::map<std::tuple<Index, Index, Index, Index, Index>, std::unique_ptr<Workspace<S>>> registry;
  std::lock_guard<std::mutex> lock(registry_mutex);
  auto& entry = registry[std::make_tuple(T, n, nx, n0, m)];
  if (!entry) {
    entry = std::make_unique<Workspace<S>>(T, n, nx, n0, m);
  }
  return *entry;
}

template <typename S>
S InfNorm(const Eigen::Ref<const Vec<S>>& v) {
  return v.size() ? v.cwiseAbs().maxCoeff() : S(0);
}

// The residuals ADMMSolver's _compute_residuals reports (projection, no slacks).
template <typename S>
void Residuals(const Problem<S>& P, const Iterate<S>& X, S* info) {
  const Index T = P.T, n = P.n, m = P.m;
  S primal = 0, cx_norm = 0, c_norm = 0, gx_norm = 0, z_norm = 0;
  S dual = 0, px_norm = 0, aty_norm = 0, q_norm = 0;
  Vec<S> Cx, Gx, Px(n), Aty(n);
  if (P.n0 > 0) {
    Cx = P.A0m() * X.xv(0);
    primal = std::max(primal, InfNorm<S>(Cx - P.c0v()));
    cx_norm = std::max(cx_norm, InfNorm<S>(Cx));
    c_norm = std::max(c_norm, InfNorm<S>(P.c0v()));
  }
  for (Index t = 0; t < T; ++t) {
    if (t + 1 < T) {
      Cx = P.Amm(t) * X.xv(t) + P.Apm(t) * X.xv(t + 1);
      primal = std::max(primal, InfNorm<S>(Cx - P.cv(t)));
      cx_norm = std::max(cx_norm, InfNorm<S>(Cx));
      c_norm = std::max(c_norm, InfNorm<S>(P.cv(t)));
    }
    if (m > 0) {
      Gx = P.Gm(t) * X.xv(t);
      primal = std::max(primal, InfNorm<S>(Gx - X.zgv(t)));
      gx_norm = std::max(gx_norm, InfNorm<S>(Gx));
      z_norm = std::max(z_norm, InfNorm<S>(X.zgv(t)));
    }
    Px.noalias() = P.Dm(t) * X.xv(t);
    if (t > 0) {
      Px.noalias() += P.Em(t - 1) * X.xv(t - 1);
    }
    if (t + 1 < T) {
      Px.noalias() += P.Em(t).transpose() * X.xv(t + 1);
    }
    Aty.setZero();
    if (t == 0 && P.n0 > 0) {
      Aty.noalias() += P.A0m().transpose() * X.yf0v();
    }
    if (t + 1 < T) {
      Aty.noalias() += P.Amm(t).transpose() * X.yfdv(t);
    }
    if (t > 0) {
      Aty.noalias() += P.Apm(t - 1).transpose() * X.yfdv(t - 1);
    }
    if (m > 0) {
      Aty.noalias() += P.Gm(t).transpose() * X.ygv(t);
    }
    dual = std::max(dual, InfNorm<S>(Px + P.qv(t) + Aty));
    px_norm = std::max(px_norm, InfNorm<S>(Px));
    aty_norm = std::max(aty_norm, InfNorm<S>(Aty));
    q_norm = std::max(q_norm, InfNorm<S>(P.qv(t)));
  }
  info[0] = primal;
  info[1] = dual;
  info[2] = std::max(std::max(cx_norm, gx_norm), std::max(c_norm, z_norm));
  info[3] = std::max(std::max(px_norm, aty_norm), q_norm);
}

template <typename S>
bool AllFinite(const S* data, Index size) {
  for (Index i = 0; i < size; ++i) {
    if (!std::isfinite(data[i])) {
      return false;
    }
  }
  return true;
}

template <typename S>
void SolveOne(const Problem<S>& P, const S* x0, const S* yg0, const S* yf00, const S* yfd0, const S* zg0, S rho,
              const Iterate<S>& X, S* info, S sigma, S rho_f_factor, S alpha, std::int64_t max_iter, int threads) {
  const Index T = P.T, n = P.n, nx = P.nx, n0 = P.n0, m = P.m;
  std::copy(x0, x0 + T * n, X.x);
  std::copy(yg0, yg0 + T * m, X.yg);
  std::copy(yf00, yf00 + n0, X.yf0);
  std::copy(yfd0, yfd0 + (T - 1) * nx, X.yfd);
  std::copy(zg0, zg0 + T * m, X.zg);
  std::fill(info, info + kInfoSize, S(0));
  const auto start = std::chrono::steady_clock::now();
  const S rho_f = rho * rho_f_factor;
  auto& ws = FindWorkspace<S>(T, n, nx, n0, m);
  std::lock_guard<std::mutex> lock(ws.mutex);
  const bool factored = ws.Factorize(P, sigma, rho, rho_f, threads);
  const auto factored_at = std::chrono::steady_clock::now();
  info[5] = factored ? S(1) : S(0);
  info[6] = ws.structured() ? (ws.negative_identity() ? S(2) : S(1)) : S(0);
  info[7] = static_cast<S>(threads);
  info[8] = static_cast<S>(std::chrono::duration<double>(factored_at - start).count());
  if (!factored) {
    const S nan = std::numeric_limits<S>::quiet_NaN();
    std::fill(X.x, X.x + T * n, nan);
    info[0] = info[1] = nan;
    info[4] = 0;
    return;
  }
  for (std::int64_t it = 0; it < max_iter; ++it) {
    ws.Iteration(P, X, sigma, rho, rho_f, alpha, threads);
  }
  info[9] = static_cast<S>(std::chrono::duration<double>(std::chrono::steady_clock::now() - factored_at).count());
  Residuals<S>(P, X, info);
  const bool finite = AllFinite(X.x, T * n) && AllFinite(X.yg, T * m) && AllFinite(X.yf0, n0) &&
                      AllFinite(X.yfd, (T - 1) * nx) && AllFinite(X.zg, T * m) && AllFinite(info, 4);
  info[4] = finite ? S(1) : S(0);
}

template <ffi::DataType DT>
using Buf = ffi::Buffer<DT>;

template <ffi::DataType DT>
ffi::Error AdmmRiccatiCpuImpl(Buf<DT> D, Buf<DT> E, Buf<DT> q, Buf<DT> A0, Buf<DT> Am, Buf<DT> Ap, Buf<DT> c0,
                              Buf<DT> c, Buf<DT> G, Buf<DT> l, Buf<DT> u, Buf<DT> x0, Buf<DT> yg0, Buf<DT> yf00,
                              Buf<DT> yfd0, Buf<DT> zg0, Buf<DT> rho, ffi::Result<Buf<DT>> x, ffi::Result<Buf<DT>> yg,
                              ffi::Result<Buf<DT>> yf0, ffi::Result<Buf<DT>> yfd, ffi::Result<Buf<DT>> zg,
                              ffi::Result<Buf<DT>> info, double sigma, double rho_f_factor, double alpha,
                              std::int64_t max_iter, std::int64_t num_threads) {
  using S = ffi::NativeType<DT>;
  // Shapes: D (B?, T, n, n), A_minus (B?, T-1, nx, n), A0 (B?, n0, n), G (B?, T, m, n).
  const auto dD = D.dimensions();
  const auto dAm = Am.dimensions();
  const auto dA0 = A0.dimensions();
  const auto dG = G.dimensions();
  if (dD.size() != 3 && dD.size() != 4) {
    return ffi::Error::InvalidArgument("admm_riccati_cpu: D must be (T, n, n) or (B, T, n, n)");
  }
  const bool batched = dD.size() == 4;
  const Index B = batched ? dD[0] : 1;
  const Index T = dD[batched ? 1 : 0];
  const Index n = dD[batched ? 2 : 1];
  const Index nx = dAm[batched ? 2 : 1];
  const Index n0 = dA0[batched ? 1 : 0];
  const Index m = dG[batched ? 2 : 1];
  if (T < 2 || n < 1 || nx < 1 || nx > n || max_iter < 0) {
    return ffi::Error::InvalidArgument("admm_riccati_cpu: needs at least 2 nodes and 0 < nx <= n");
  }
  auto size = [](const auto& buffer) { return static_cast<Index>(buffer.element_count()); };
  if (size(Am) != B * (T - 1) * nx * n || size(Ap) != B * (T - 1) * nx * n || size(E) != B * (T - 1) * n * n ||
      size(q) != B * T * n || size(c) != B * (T - 1) * nx || size(c0) != B * n0 || size(G) != B * T * m * n ||
      size(l) != B * T * m || size(u) != B * T * m || size(x0) != B * T * n || size(yg0) != B * T * m ||
      size(zg0) != B * T * m || size(yf00) != B * n0 || size(yfd0) != B * (T - 1) * nx || size(rho) != B ||
      size(*info) != B * kInfoSize) {
    return ffi::Error::InvalidArgument("admm_riccati_cpu: inconsistent buffer sizes");
  }
  const int threads = static_cast<int>(std::max<std::int64_t>(1, std::min<std::int64_t>(num_threads, 64)));
  try {
    for (Index b = 0; b < B; ++b) {
      Problem<S> P{T, n, nx, n0, m,
                   D.typed_data() + b * T * n * n,
                   E.typed_data() + b * (T - 1) * n * n,
                   q.typed_data() + b * T * n,
                   A0.typed_data() + b * n0 * n,
                   Am.typed_data() + b * (T - 1) * nx * n,
                   Ap.typed_data() + b * (T - 1) * nx * n,
                   c0.typed_data() + b * n0,
                   c.typed_data() + b * (T - 1) * nx,
                   G.typed_data() + b * T * m * n,
                   l.typed_data() + b * T * m,
                   u.typed_data() + b * T * m};
      Iterate<S> X{x->typed_data() + b * T * n, yg->typed_data() + b * T * m, yf0->typed_data() + b * n0,
                   yfd->typed_data() + b * (T - 1) * nx, zg->typed_data() + b * T * m, T, n, nx, n0, m};
      SolveOne<S>(P, x0.typed_data() + b * T * n, yg0.typed_data() + b * T * m, yf00.typed_data() + b * n0,
                  yfd0.typed_data() + b * (T - 1) * nx, zg0.typed_data() + b * T * m, rho.typed_data()[b], X,
                  info->typed_data() + b * kInfoSize, static_cast<S>(sigma), static_cast<S>(rho_f_factor),
                  static_cast<S>(alpha), max_iter, threads);
    }
  } catch (const std::exception& error) {
    return ffi::Error::Internal(error.what());
  }
  return ffi::Error::Success();
}

template <ffi::DataType DT>
auto Bind() {
  return ffi::Ffi::Bind()
      .template Arg<Buf<DT>>()  // D
      .template Arg<Buf<DT>>()  // E
      .template Arg<Buf<DT>>()  // q
      .template Arg<Buf<DT>>()  // A0
      .template Arg<Buf<DT>>()  // A_minus
      .template Arg<Buf<DT>>()  // A_plus
      .template Arg<Buf<DT>>()  // c0
      .template Arg<Buf<DT>>()  // c
      .template Arg<Buf<DT>>()  // G
      .template Arg<Buf<DT>>()  // l
      .template Arg<Buf<DT>>()  // u
      .template Arg<Buf<DT>>()  // x0
      .template Arg<Buf<DT>>()  // y_g
      .template Arg<Buf<DT>>()  // y_f_0
      .template Arg<Buf<DT>>()  // y_f_dyn
      .template Arg<Buf<DT>>()  // z_g
      .template Arg<Buf<DT>>()  // rho
      .template Ret<Buf<DT>>()  // x
      .template Ret<Buf<DT>>()  // y_g
      .template Ret<Buf<DT>>()  // y_f_0
      .template Ret<Buf<DT>>()  // y_f_dyn
      .template Ret<Buf<DT>>()  // z_g
      .template Ret<Buf<DT>>()  // info
      .template Attr<double>("sigma")
      .template Attr<double>("rho_f_factor")
      .template Attr<double>("alpha")
      .template Attr<std::int64_t>("max_iter")
      .template Attr<std::int64_t>("num_threads");
}

}  // namespace

extern "C" XLA_FFI_Error* AdmmRiccatiCpu(XLA_FFI_CallFrame* call_frame) {
  static auto* handler = Bind<ffi::DataType::F32>().To(AdmmRiccatiCpuImpl<ffi::DataType::F32>).release();
  return handler->Call(call_frame);
}

extern "C" XLA_FFI_Error* AdmmRiccatiCpuF64(XLA_FFI_CallFrame* call_frame) {
  static auto* handler = Bind<ffi::DataType::F64>().To(AdmmRiccatiCpuImpl<ffi::DataType::F64>).release();
  return handler->Call(call_frame);
}
