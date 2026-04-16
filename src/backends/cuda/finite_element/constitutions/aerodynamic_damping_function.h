#pragma once
#include <type_define.h>

namespace uipc::backend::cuda
{
namespace sym::aerodynamic_damping
{
/**
 * Aerodynamic damping energy for a single triangle.
 *
 * Models air drag as: E = coeff * dot(u, c)^2 / (18 * ||c||)
 *
 * where:
 *   c = (x1 - x0) x (x2 - x0)   (cross product = 2 * area * normal)
 *   u = (x0 + x1 + x2) - (x0p + x1p + x2p)  (displacement sum)
 *   coeff = drag_coefficient (combines C_d * rho_air / 2)
 *
 * This is the incremental potential for Rayleigh dissipation
 * D = 0.5 * coeff * A * v_n^2 where v_n is the face-normal velocity.
 */

// 3x3 skew-symmetric matrix for cross product: [v]_x * w = v x w
inline UIPC_GENERIC Matrix3x3 skew(const Vector3& v)
{
    Matrix3x3 S;
    S << 0, -v(2), v(1),
         v(2), 0, -v(0),
         -v(1), v(0), 0;
    return S;
}

// Energy
inline UIPC_GENERIC Float E(const Vector3& x0, const Vector3& x1, const Vector3& x2,
                             const Vector3& x0p, const Vector3& x1p, const Vector3& x2p,
                             Float coeff)
{
    Vector3 a = x1 - x0;
    Vector3 b = x2 - x0;
    Vector3 c = a.cross(b);
    Float cn = c.norm();
    if(cn < 1e-20)
        return 0.0;

    Vector3 u = (x0 + x1 + x2) - (x0p + x1p + x2p);
    Float alpha = u.dot(c);

    // E = coeff / 18 * alpha^2 / cn
    return coeff / 18.0 * alpha * alpha / cn;
}

// Gradient: dE/d[x0, x1, x2] (9-vector)
inline UIPC_GENERIC void dEdx(Vector<Float, 9>& G,
                               const Vector3& x0, const Vector3& x1, const Vector3& x2,
                               const Vector3& x0p, const Vector3& x1p, const Vector3& x2p,
                               Float coeff)
{
    Vector3 a = x1 - x0;
    Vector3 b = x2 - x0;
    Vector3 c = a.cross(b);
    Float cn = c.norm();

    if(cn < 1e-20)
    {
        G.setZero();
        return;
    }

    Vector3 n = c / cn;
    Vector3 u = (x0 + x1 + x2) - (x0p + x1p + x2p);
    Float alpha = u.dot(c);
    Float k = coeff / 18.0;

    // dc/dx0 = [x2-x1]_x,  dc/dx1 = -[x2-x0]_x = -[b]_x,  dc/dx2 = [x1-x0]_x = [a]_x
    Matrix3x3 dc_dx0 = skew(x2 - x1);
    Matrix3x3 dc_dx1 = -skew(b);
    Matrix3x3 dc_dx2 = skew(a);

    // d(alpha)/dx_i = c + dc_dx_i^T * u  (since du/dx_i = I for all i)
    Vector3 dalpha_dx0 = c + dc_dx0.transpose() * u;
    Vector3 dalpha_dx1 = c + dc_dx1.transpose() * u;
    Vector3 dalpha_dx2 = c + dc_dx2.transpose() * u;

    // d(cn)/dx_i = n^T * dc_dx_i  (row vector, transposed to column)
    Vector3 dcn_dx0 = dc_dx0.transpose() * n;
    Vector3 dcn_dx1 = dc_dx1.transpose() * n;
    Vector3 dcn_dx2 = dc_dx2.transpose() * n;

    // dE/dx_i = k * (2*alpha/cn * dalpha_dx_i - alpha^2/cn^2 * dcn_dx_i)
    Float f1 = 2.0 * k * alpha / cn;
    Float f2 = k * alpha * alpha / (cn * cn);

    G.segment<3>(0) = f1 * dalpha_dx0 - f2 * dcn_dx0;
    G.segment<3>(3) = f1 * dalpha_dx1 - f2 * dcn_dx1;
    G.segment<3>(6) = f1 * dalpha_dx2 - f2 * dcn_dx2;
}

// Hessian: d^2E/dx^2 (9x9 matrix) — uses outer-product approximation + SPD guarantee
inline UIPC_GENERIC void ddEddx(Matrix<Float, 9, 9>& H,
                                  const Vector3& x0, const Vector3& x1, const Vector3& x2,
                                  const Vector3& x0p, const Vector3& x1p, const Vector3& x2p,
                                  Float coeff)
{
    Vector3 a = x1 - x0;
    Vector3 b = x2 - x0;
    Vector3 c = a.cross(b);
    Float cn = c.norm();

    if(cn < 1e-20)
    {
        H.setZero();
        return;
    }

    Vector3 n = c / cn;
    Vector3 u = (x0 + x1 + x2) - (x0p + x1p + x2p);
    Float alpha = u.dot(c);
    Float k = coeff / 18.0;

    // Recompute gradient components
    Matrix3x3 dc_dx0 = skew(x2 - x1);
    Matrix3x3 dc_dx1 = -skew(b);
    Matrix3x3 dc_dx2 = skew(a);

    Vector3 dalpha_dx0 = c + dc_dx0.transpose() * u;
    Vector3 dalpha_dx1 = c + dc_dx1.transpose() * u;
    Vector3 dalpha_dx2 = c + dc_dx2.transpose() * u;

    // Gauss-Newton approximation: H ≈ 2*k/cn * g * g^T
    // where g = [dalpha_dx0; dalpha_dx1; dalpha_dx2]
    // This is guaranteed PSD and is the same for both biased and unbiased energy.
    Vector<Float, 9> g;
    g.segment<3>(0) = dalpha_dx0;
    g.segment<3>(3) = dalpha_dx1;
    g.segment<3>(6) = dalpha_dx2;

    H = (2.0 * k / cn) * g * g.transpose();
}

}  // namespace sym::aerodynamic_damping
}  // namespace uipc::backend::cuda
