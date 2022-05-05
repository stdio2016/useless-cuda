clang -fmodules -fobjc-arc -framework CoreGraphics metal.m -O3 -o metal
xcrun -sdk macosx metal -c kernel.metal -o kernel.air
xcrun -sdk macosx metallib kernel.air   -o kernel.metallib
