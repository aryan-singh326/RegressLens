// Runtime dispatch and kernel-set parity. Compiled WITHOUT AVX2 flags:
// it reaches the AVX2 kernels only through the dispatcher, exactly as
// the Python extension does, so it runs (and is meaningful) on any CPU.
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

#include "regresslens/cpu_features.hpp"
#include "regresslens/dispatch.hpp"
#include "regresslens/kernels.hpp"

#include "test_util.hpp"

using namespace regresslens;

static void test_runtime_matches_cpu() {
    const Runtime& rt = runtime();
    CHECK(rt.error.empty());
    CHECK(rt.cpu_avx2 == cpu_has_avx2_fma());
    CHECK((rt.avx2 != nullptr) == (rt.avx2_compiled && rt.cpu_avx2));
    std::printf("test_runtime_matches_cpu passed (avx2_compiled=%d cpu_avx2=%d)\n",
                rt.avx2_compiled, rt.cpu_avx2);
}

static void test_resolve_kernel() {
    SelectionContext ctx{Operation::Reduction, DType::Float64, 1000, true, 8};
    CHECK(resolve_kernel(ctx, "scalar", true) == KernelChoice::Scalar);
    bool threw = false;
    try {
        resolve_kernel(ctx, "sse9", true);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    KernelChoice autochoice = resolve_kernel(ctx, "auto", true);
    if (runtime().avx2 == nullptr) CHECK(autochoice == KernelChoice::Scalar);
    // Projection has no MT kernel; an MT selection must degrade.
    SelectionContext proj{Operation::Projection, DType::Float64, 100'000'000, true, 8};
    CHECK(resolve_kernel(proj, "auto", false) != KernelChoice::MtAvx2);
    std::printf("test_resolve_kernel passed\n");
}

template <typename T>
static void check_table_parity(const OpTable<T>& a, const OpTable<T>& b, const char* label) {
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    for (size_t n : {size_t(1), size_t(7), size_t(33), size_t(1000), size_t(65'537)}) {
        std::vector<T> in(n);
        for (auto& v : in) v = static_cast<T>(dist(rng));

        SumParts pa = a.sum(in.data(), n), pb = b.sum(in.data(), n);
        double sa = pa.s + pa.c, sb = pb.s + pb.c;
        CHECK(std::fabs(sa - sb) <= 1e-14 * std::max(1.0, std::fabs(sa)));

        std::vector<T> oa(n), ob(n);
        size_t ca = a.filter_gt(in.data(), oa.data(), n, T(0));
        size_t cb = b.filter_gt(in.data(), ob.data(), n, T(0));
        CHECK(ca == cb);
        for (size_t i = 0; i < ca; ++i) CHECK(oa[i] == ob[i]);

        a.project_affine(in.data(), oa.data(), n, T(2.5), T(-1.25));
        b.project_affine(in.data(), ob.data(), n, T(2.5), T(-1.25));
        for (size_t i = 0; i < n; ++i) {
            // FMA (one rounding) vs multiply-then-add (two): at most a
            // couple of ulps apart.
            double tol = 4 * std::numeric_limits<T>::epsilon() *
                         std::max(1.0, std::fabs(static_cast<double>(oa[i])));
            CHECK(std::fabs(static_cast<double>(oa[i]) - static_cast<double>(ob[i])) <= tol);
        }

        size_t window = 1 + n / 3;
        size_t outs = n - window + 1;
        a.rolling_sum(in.data(), oa.data(), n, window);
        b.rolling_sum(in.data(), ob.data(), n, window);
        for (size_t i = 0; i < outs; ++i) {
            double tol = 2 * std::numeric_limits<T>::epsilon() *
                         std::max(1.0, std::fabs(static_cast<double>(oa[i])));
            CHECK(std::fabs(static_cast<double>(oa[i]) - static_cast<double>(ob[i])) <= tol);
        }

        ScanResult ra = a.scan(in.data(), n), rb = b.scan(in.data(), n);
        CHECK(ra.nonfinite == rb.nonfinite && ra.absmax == rb.absmax);
    }
    std::printf("%s table parity passed\n", label);
}

int main() {
    test_runtime_matches_cpu();
    test_resolve_kernel();
    const KernelSet& scalar = scalar_kernel_set();
    CHECK(scalar.f64.sum_mt == nullptr);
    if (runtime().avx2 != nullptr) {
        check_table_parity(scalar.f64, runtime().avx2->f64, "f64 scalar-vs-avx2");
        check_table_parity(scalar.f32, runtime().avx2->f32, "f32 scalar-vs-avx2");
    } else {
        std::printf("no AVX2 kernel set on this CPU/build; parity checks skipped\n");
    }
    std::printf("All dispatch tests passed.\n");
    return 0;
}
