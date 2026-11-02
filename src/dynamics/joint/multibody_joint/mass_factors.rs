//! A jointed body's mass matrix, factored along its joint tree.

use crate::alloc_prelude::*;
use crate::math::{DMatrix, Real};
use na::{Dyn, LU, Matrix, StorageMut, U1};

/// The parent of a degree of freedom that has none.
pub(crate) const ROOT: usize = usize::MAX;

/// The factors that solve with a jointed body's mass matrix.
///
/// Along the joint tree, `M = Lᵀ·D·L` with `L` unit lower triangular: the
/// LTDL factorization of Featherstone 2005, "Efficient factorization of the
/// joint-space inertia matrix for branched kinematic trees" (*Rigid Body
/// Dynamics Algorithms* §6.5). Two degrees of freedom couple only where one
/// moves with the other, so eliminating the leaves first fills nothing in,
/// and a solve walks each degree's ancestors where a dense one sweeps every
/// row. A body some of whose degrees move kinematically, left out of its
/// matrix, is factored densely, as is one whose elimination meets a
/// vanishing pivot.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub enum MassFactors {
    /// Along the joint tree.
    Tree {
        /// Degree `k`'s ancestors are `ancestors[starts[k]..starts[k + 1]]`,
        /// its parent first: an ancestor's own follow it in the list.
        starts: Vec<usize>,
        ancestors: Vec<usize>,
        /// `L[k][i]` for each degree `k` and ancestor `i`, in `ancestors`' order.
        lower: Vec<Real>,
        /// `1 / D[k][k]`.
        inv_pivots: Vec<Real>,
    },
    /// Dense, with partial pivoting.
    Dense(LU<Real, Dyn, Dyn>),
}

impl Default for MassFactors {
    fn default() -> Self {
        MassFactors::Dense(LU::new(DMatrix::zeros(0, 0)))
    }
}

impl MassFactors {
    /// Factors the first `dim` rows and columns of the symmetric `matrix`:
    /// along the tree `parents` describes, when it describes one, or else
    /// densely.
    pub(crate) fn of(matrix: &DMatrix, dim: usize, parents: Option<&[usize]>) -> Self {
        parents
            .and_then(|parents| Self::along_tree(matrix, parents))
            .unwrap_or_else(|| {
                MassFactors::Dense(LU::new(matrix.view((0, 0), (dim, dim)).into_owned()))
            })
    }

    /// Eliminates the leaves first (*RBDA* table 6.3): each degree's row
    /// folds into its ancestors', whose own ancestors are the rest of it.
    fn along_tree(a: &DMatrix, parents: &[usize]) -> Option<Self> {
        let n = parents.len();
        let mut starts = Vec::with_capacity(n + 1);
        starts.push(0);
        for (k, &parent) in parents.iter().enumerate() {
            let depth = if parent == ROOT { 0 } else { starts[parent + 1] - starts[parent] + 1 };
            starts.push(starts[k] + depth);
        }
        let mut ancestors = Vec::with_capacity(starts[n]);
        let mut lower = Vec::with_capacity(starts[n]);
        for (k, &parent) in parents.iter().enumerate() {
            if parent != ROOT {
                ancestors.push(parent);
                ancestors.extend_from_within(starts[parent]..starts[parent + 1]);
            }
            lower.extend(ancestors[starts[k]..].iter().map(|&i| a[(k, i)]));
        }
        let mut pivots: Vec<Real> = (0..n).map(|k| a[(k, k)]).collect();
        let tiny = Real::EPSILON * pivots.iter().fold(0.0, |most: Real, p| most.max(p.abs()));
        for k in (0..n).rev() {
            let pivot = pivots[k];
            if !pivot.is_finite() || pivot.abs() <= tiny {
                return None;
            }
            let (before, row) = lower.split_at_mut(starts[k]);
            let row = &mut row[..starts[k + 1] - starts[k]];
            for m in 0..row.len() {
                let i = ancestors[starts[k] + m];
                let share = row[m] / pivot;
                pivots[i] -= share * row[m];
                for (x, &y) in before[starts[i]..starts[i + 1]].iter_mut().zip(&row[m + 1..]) {
                    *x -= share * y;
                }
                row[m] = share;
            }
        }
        pivots.iter_mut().for_each(|pivot| *pivot = 1.0 / *pivot);
        Some(MassFactors::Tree {
            starts,
            ancestors,
            lower,
            inv_pivots: pivots,
        })
    }

    /// Solves `M·x = b` in place; `false` where dense factors are singular.
    pub fn solve_mut<S: StorageMut<Real, Dyn, U1>>(&self, b: &mut Matrix<Real, Dyn, U1, S>) -> bool {
        match self {
            MassFactors::Dense(lu) => lu.solve_mut(b),
            MassFactors::Tree {
                starts,
                ancestors,
                lower,
                inv_pivots,
            } => {
                let n = inv_pivots.len();
                // `Lᵀ·z = b`, from the leaves: a settled degree passes its
                // share on to its ancestors.
                for k in (0..n).rev() {
                    let settled = b[k];
                    if settled != 0.0 {
                        let at = starts[k]..starts[k + 1];
                        for (&i, &share) in ancestors[at.clone()].iter().zip(&lower[at]) {
                            b[i] -= share * settled;
                        }
                    }
                }
                // `D·y = z`, and `L·x = y` from the root: each degree takes its
                // ancestors' share.
                for k in 0..n {
                    let at = starts[k]..starts[k + 1];
                    let mut x = b[k] * inv_pivots[k];
                    for (&i, &share) in ancestors[at.clone()].iter().zip(&lower[at]) {
                        x -= share * b[i];
                    }
                    b[k] = x;
                }
                true
            }
        }
    }
}
