use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlackDevice {
    BlackCpu,
    BlackCuda(u32),
    BlackMetal(u32),
}

impl BlackDevice {
    pub fn black_is_cpu(&self) -> bool {
        matches!(self, BlackDevice::BlackCpu)
    }

    pub fn black_is_cuda(&self) -> bool {
        matches!(self, BlackDevice::BlackCuda(_))
    }

    pub fn black_is_metal(&self) -> bool {
        matches!(self, BlackDevice::BlackMetal(_))
    }

    pub fn black_ordinal(&self) -> Option<u32> {
        match self {
            BlackDevice::BlackCpu => None,
            BlackDevice::BlackCuda(black_id) => Some(*black_id),
            BlackDevice::BlackMetal(black_id) => Some(*black_id),
        }
    }
}


impl std::fmt::Display for BlackDevice {
    fn fmt(&self, black_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlackDevice::BlackCpu => write!(black_f, "black_cpu"),
            BlackDevice::BlackCuda(black_id) => write!(black_f, "black_cuda:{}", black_id),
            BlackDevice::BlackMetal(black_id) => write!(black_f, "black_metal:{}", black_id),
        }
    }
}
