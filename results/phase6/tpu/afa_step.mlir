module @jit_afa_step attributes {mhlo.num_partitions = 16 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<16x4x256x64xf32> {mhlo.sharding = "{devices=[16,1,1,1]<=[16]}"}, %arg1: tensor<16x4x256x64xf32> {mhlo.sharding = "{devices=[16,1,1,1]<=[16]}"}, %arg2: tensor<16x4x256x64xf32> {mhlo.sharding = "{devices=[16,1,1,1]<=[16]}"}) -> (tensor<16x4x256x64xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[16,1,1,1]<=[16]}"} : (tensor<16x4x256x64xf32>) -> tensor<16x4x256x64xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<16x4x256x64xf32>) -> tensor<1x4x256x64xf32>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[16,1,1,1]<=[16]}"} : (tensor<16x4x256x64xf32>) -> tensor<16x4x256x64xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<16x4x256x64xf32>) -> tensor<1x4x256x64xf32>
    %4 = stablehlo.custom_call @Sharding(%arg2) {backend_config = "", mhlo.sharding = "{devices=[16,1,1,1]<=[16]}"} : (tensor<16x4x256x64xf32>) -> tensor<16x4x256x64xf32>
    %5 = stablehlo.custom_call @SPMDFullToShardShape(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<16x4x256x64xf32>) -> tensor<1x4x256x64xf32>
    %6 = call @shmap_body(%1, %3, %5) : (tensor<1x4x256x64xf32>, tensor<1x4x256x64xf32>, tensor<1x4x256x64xf32>) -> tensor<1x4x256x64xf32>
    %7 = stablehlo.custom_call @Sharding(%6) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x4x256x64xf32>) -> tensor<1x4x256x64xf32>
    %8 = stablehlo.custom_call @SPMDShardToFullShape(%7) {backend_config = "", mhlo.sharding = "{devices=[16,1,1,1]<=[16]}"} : (tensor<1x4x256x64xf32>) -> tensor<16x4x256x64xf32>
    return %8 : tensor<16x4x256x64xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x4x256x64xf32>, %arg1: tensor<1x4x256x64xf32>, %arg2: tensor<1x4x256x64xf32>) -> (tensor<1x4x256x64xf32> {jax.result_info = "[('d',), None, None, None]"}) {
    %cst = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4x256x64xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1x4x256x64xf32>
    %2 = stablehlo.transpose %arg1, dims = [0, 1, 3, 2] : (tensor<1x4x256x64xf32>) -> tensor<1x4x64x256xf32>
    %3 = stablehlo.reshape %1 : (tensor<1x4x256x64xf32>) -> tensor<4x256x64xf32>
    %4 = stablehlo.dot_general %3, %2, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [HIGHEST, HIGHEST] : (tensor<4x256x64xf32>, tensor<1x4x64x256xf32>) -> tensor<4x256x1x256xf32>
    %5 = stablehlo.transpose %4, dims = [2, 0, 1, 3] : (tensor<4x256x1x256xf32>) -> tensor<1x4x256x256xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<1x4x256x256xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x4x256x256xf32>
    %8 = stablehlo.add %7, %6 : tensor<1x4x256x256xf32>
    %9 = stablehlo.rsqrt %8 : tensor<1x4x256x256xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<1x4x256x256xf32>
    %11 = stablehlo.add %5, %10 : tensor<1x4x256x256xf32>
    %12 = stablehlo.multiply %11, %11 : tensor<1x4x256x256xf32>
    %13 = stablehlo.multiply %12, %12 : tensor<1x4x256x256xf32>
    %14 = stablehlo.multiply %13, %13 : tensor<1x4x256x256xf32>
    %15 = stablehlo.reshape %14 : (tensor<1x4x256x256xf32>) -> tensor<4x256x256xf32>
    %16 = stablehlo.dot_general %15, %arg2, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [HIGHEST, HIGHEST] : (tensor<4x256x256xf32>, tensor<1x4x256x64xf32>) -> tensor<4x256x1x64xf32>
    %17 = stablehlo.transpose %16, dims = [2, 0, 1, 3] : (tensor<4x256x1x64xf32>) -> tensor<1x4x256x64xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %18 = stablehlo.reduce(%14 init: %cst_1) applies stablehlo.add across dimensions = [3] : (tensor<1x4x256x256xf32>, tensor<f32>) -> tensor<1x4x256xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1, 2] : (tensor<1x4x256xf32>) -> tensor<1x4x256x1xf32>
    %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x4x256x1xf32>
    %21 = stablehlo.add %19, %20 : tensor<1x4x256x1xf32>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x4x256x1xf32>
    %23 = stablehlo.divide %22, %21 : tensor<1x4x256x1xf32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [0, 1, 2, 3] : (tensor<1x4x256x1xf32>) -> tensor<1x4x256x64xf32>
    %25 = stablehlo.multiply %17, %24 : tensor<1x4x256x64xf32>
    return %25 : tensor<1x4x256x64xf32>
  }
}
