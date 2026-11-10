use std::hint::black_box;
use upodesh::corrector;

fn main() {
    let c = corrector();
    for _ in 0..2000 {
        black_box(c.suggest_with("বাংরা", 10));
    }
}
