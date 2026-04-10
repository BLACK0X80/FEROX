use crate::black_error::{BlackError, BlackResult};
use serde::{Deserialize, Serialize};

const BLACK_MAX_STACK_RANK: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackShape {
    black_dims_stack: [usize; BLACK_MAX_STACK_RANK],
    black_dims_heap: Option<Vec<usize>>,
    black_rank: usize,
}

impl BlackShape {
    pub fn black_new(black_dims: &[usize]) -> Self {
        let black_rank = black_dims.len();
        if black_rank <= BLACK_MAX_STACK_RANK {
            let mut black_dims_stack = [0usize; BLACK_MAX_STACK_RANK];
            black_dims_stack[..black_rank].copy_from_slice(black_dims);
            BlackShape {
                black_dims_stack,
                black_dims_heap: None,
                black_rank,
            }
        } else {
            BlackShape {
                black_dims_stack: [0usize; BLACK_MAX_STACK_RANK],
                black_dims_heap: Some(black_dims.to_vec()),
                black_rank,
            }
        }
    }

    pub fn black_scalar() -> Self {
        BlackShape {
            black_dims_stack: [0usize; BLACK_MAX_STACK_RANK],
            black_dims_heap: None,
            black_rank: 0,
        }
    }

    pub fn black_ndim(&self) -> usize {
        self.black_rank
    }

    pub fn black_dims(&self) -> &[usize] {
        if let Some(ref black_heap) = self.black_dims_heap {
            black_heap.as_slice()
        } else {
            &self.black_dims_stack[..self.black_rank]
        }
    }

    pub fn black_dim(&self, black_idx: usize) -> BlackResult<usize> {
        let black_dims = self.black_dims();
        if black_idx >= self.black_rank {
            return Err(BlackError::BlackIndexError {
                black_msg: format!(
                    "dimension index {} out of range for rank {}",
                    black_idx, self.black_rank
                ),
            });
        }
        Ok(black_dims[black_idx])
    }

    pub fn black_numel(&self) -> usize {
        if self.black_rank == 0 {
            return 1;
        }
        self.black_dims().iter().product()
    }

    pub fn black_broadcast_with(&self, black_other: &BlackShape) -> BlackResult<BlackShape> {
        let black_max_rank = self.black_rank.max(black_other.black_rank);
        let mut black_result_dims = vec![0usize; black_max_rank];
        let black_a = self.black_dims();
        let black_b = black_other.black_dims();

        for black_i in 0..black_max_rank {
            let black_da = if black_i < self.black_rank {
                black_a[self.black_rank - 1 - black_i]
            } else {
                1
            };
            let black_db = if black_i < black_other.black_rank {
                black_b[black_other.black_rank - 1 - black_i]
            } else {
                1
            };

            if black_da == black_db {
                black_result_dims[black_max_rank - 1 - black_i] = black_da;
            } else if black_da == 1 {
                black_result_dims[black_max_rank - 1 - black_i] = black_db;
            } else if black_db == 1 {
                black_result_dims[black_max_rank - 1 - black_i] = black_da;
            } else {
                return Err(BlackError::BlackShapeError {
                    black_msg: format!(
                        "cannot broadcast shapes {:?} and {:?}",
                        self.black_dims(),
                        black_other.black_dims()
                    ),
                });
            }
        }

        Ok(BlackShape::black_new(&black_result_dims))
    }
}

impl PartialEq for BlackShape {
    fn eq(&self, black_other: &Self) -> bool {
        self.black_dims() == black_other.black_dims()
    }
}

impl Eq for BlackShape {}

impl std::fmt::Display for BlackShape {
    fn fmt(&self, black_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(black_f, "BlackShape({:?})", self.black_dims())
    }
}
