use std::hint::black_box;
use upodesh::corrector;
use peak_alloc::PeakAlloc;

#[global_allocator]
static PEAK_ALLOC: PeakAlloc = PeakAlloc;

fn main() {
    let c = corrector();
    for _ in 0..2000 {
        black_box(c.suggest_with("বাংরা", 10));
    }
    
    let current_mem = PEAK_ALLOC.current_usage_as_mb();
	println!("\nThis program currently uses {} MB of RAM.", current_mem);
	let peak_mem = PEAK_ALLOC.peak_usage_as_mb();
	println!("The max amount that was used {} MB of RAM.", peak_mem);
}
