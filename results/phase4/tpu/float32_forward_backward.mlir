module @jit_oace_both attributes {mhlo.num_partitions = 16 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<256x1024xf32> {mhlo.sharding = "{devices=[16,1]<=[16]}"}, %arg1: tensor<256xi32> {mhlo.sharding = "{devices=[16]<=[16]}"}, %arg2: tensor<256xf32> {mhlo.sharding = "{devices=[16]<=[16]}"}) -> (tensor<256xf32> {jax.result_info = "result[0]"}, tensor<256x1024xf32> {jax.result_info = "result[1]"}) {
    %0:4 = call @oace_loss(%arg0, %arg1) : (tensor<256x1024xf32>, tensor<256xi32>) -> (tensor<256xf32>, tensor<256x1024xf32>, tensor<256x1024xf32>, tensor<256x1024xf32>)
    %1 = call @oace_loss_0(%0#1, %0#2, %0#3, %arg2) : (tensor<256x1024xf32>, tensor<256x1024xf32>, tensor<256x1024xf32>, tensor<256xf32>) -> tensor<256x1024xf32>
    return %0#0, %1 : tensor<256xf32>, tensor<256x1024xf32>
  }
  func.func private @oace_loss(%arg0: tensor<256x1024xf32>, %arg1: tensor<256xi32>) -> (tensor<256xf32>, tensor<256x1024xf32>, tensor<256x1024xf32>, tensor<256x1024xf32>) {
    %0 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<256xi32>) -> tensor<256x1xi32>
    %2 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<1024xi32>) -> tensor<1x1024xi32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x1024xi32>) -> tensor<256x1024xi32>
    %4 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<256x1xi32>) -> tensor<256x1024xi32>
    %5 = stablehlo.compare  EQ, %3, %4,  SIGNED : (tensor<256x1024xi32>, tensor<256x1024xi32>) -> tensor<256x1024xi1>
    %6 = stablehlo.convert %5 : (tensor<256x1024xi1>) -> tensor<256x1024xf32>
    %7 = stablehlo.rsqrt %arg0 : tensor<256x1024xf32>
    %8 = stablehlo.rsqrt %7 : tensor<256x1024xf32>
    %9 = stablehlo.rsqrt %8 : tensor<256x1024xf32>
    %10 = stablehlo.multiply %6, %9 : tensor<256x1024xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.reduce(%10 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<256x1024xf32>, tensor<f32>) -> tensor<256xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %13 = stablehlo.subtract %11, %12 : tensor<256xf32>
    %cst_1 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %15 = stablehlo.multiply %14, %13 : tensor<256xf32>
    return %15, %6, %7, %9 : tensor<256xf32>, tensor<256x1024xf32>, tensor<256x1024xf32>, tensor<256x1024xf32>
  }
  func.func private @oace_loss_0(%arg0: tensor<256x1024xf32>, %arg1: tensor<256x1024xf32>, %arg2: tensor<256x1024xf32>, %arg3: tensor<256xf32>) -> tensor<256x1024xf32> {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<256xf32>) -> tensor<256x1xf32>
    %cst = stablehlo.constant dense<-2.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x1024xf32>
    %2 = stablehlo.multiply %1, %arg0 : tensor<256x1024xf32>
    %3 = stablehlo.multiply %2, %arg2 : tensor<256x1024xf32>
    %4 = stablehlo.multiply %arg1, %arg1 : tensor<256x1024xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<256x1024xf32>
    %6 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<256x1xf32>) -> tensor<256x1024xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<256x1024xf32>
    return %7 : tensor<256x1024xf32>
  }
}
