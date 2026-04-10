use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlackDType {
    BlackF16,
    BlackBF16,
    BlackF32,
    BlackF64,
    BlackI8,
    BlackI16,
    BlackI32,
    BlackI64,
    BlackU8,
    BlackBool,
}

impl BlackDType {
    pub fn black_size_in_bytes(&self) -> usize {
        match self {
            BlackDType::BlackF16 => 2,
            BlackDType::BlackBF16 => 2,
            BlackDType::BlackF32 => 4,
            BlackDType::BlackF64 => 8,
            BlackDType::BlackI8 => 1,
            BlackDType::BlackI16 => 2,
            BlackDType::BlackI32 => 4,
            BlackDType::BlackI64 => 8,
            BlackDType::BlackU8 => 1,
            BlackDType::BlackBool => 1,
        }
    }

    pub fn black_is_float(&self) -> bool {
        matches!(
            self,
            BlackDType::BlackF16 | BlackDType::BlackBF16 | BlackDType::BlackF32 | BlackDType::BlackF64
        )
    }

    pub fn black_is_int(&self) -> bool {
        matches!(
            self,
            BlackDType::BlackI8
                | BlackDType::BlackI16
                | BlackDType::BlackI32
                | BlackDType::BlackI64
                | BlackDType::BlackU8
        )
    }
}

impl std::fmt::Display for BlackDType {
    fn fmt(&self, black_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let black_name = match self {
            BlackDType::BlackF16 => "black_f16",
            BlackDType::BlackBF16 => "black_bf16",
            BlackDType::BlackF32 => "black_f32",
            BlackDType::BlackF64 => "black_f64",
            BlackDType::BlackI8 => "black_i8",
            BlackDType::BlackI16 => "black_i16",
            BlackDType::BlackI32 => "black_i32",
            BlackDType::BlackI64 => "black_i64",
            BlackDType::BlackU8 => "black_u8",
            BlackDType::BlackBool => "black_bool",
        };
        write!(black_f, "{}", black_name)
    }
}
