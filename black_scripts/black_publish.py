import subprocess
import os

def black_build_and_publish():
    subprocess.check_call(["python", "-m", "pip", "install", "maturin", "twine", "build"])
    
    subprocess.check_call(
        ["python", "-m", "maturin", "build", "--release", "--out", "dist", "-m", "black_bind/Cargo.toml"]
    )
    subprocess.check_call(
        ["python", "-m", "maturin", "sdist", "--out", "dist", "-m", "black_bind/Cargo.toml"]
    )
    
    # twine upload dist/*
    black_dist_files = os.listdir("dist")
    print(f"black_built_files: {black_dist_files}")
    print("black_publish_ready")

if __name__ == "__main__":
    black_build_and_publish()
