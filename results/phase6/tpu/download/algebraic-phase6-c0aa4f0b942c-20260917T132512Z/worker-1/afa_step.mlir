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
    %0 = stablehlo.transpose %arg1, dims = [0, 1, 3, 2] : (tensor<1x4x256x64xf32>) -> tensor<1x4x64x256xf32>
    %1 = stablehlo.reshape %arg0 : (tensor<1x4x256x64xf32>) -> tensor<4x256x64xf32>
    %2 = stablehlo.dot_general %1, %0, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [HIGHEST, HIGHEST] : (tensor<4x256x64xf32>, tensor<1x4x64x256xf32>) -> tensor<4x256x1x256xf32>
    %3 = stablehlo.transpose %2, dims = [2, 0, 1, 3] : (tensor<4x256x1x256xf32>) -> tensor<1x4x256x256xf32>
    %cst = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4x256x256xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1x4x256x256xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<1x4x256x256xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x4x256x256xf32>
    %8 = stablehlo.add %7, %6 : tensor<1x4x256x256xf32>
    %9 = stablehlo.rsqrt %8 : tensor<1x4x256x256xf32>
    %10 = stablehlo.multiply %5, %9 : tensor<1x4x256x256xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x4x256x256xf32>
    %12 = stablehlo.compare  LT, %5, %11,  FLOAT : (tensor<1x4x256x256xf32>, tensor<1x4x256x256xf32>) -> tensor<1x4x256x256xi1>
    %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x4x256x256xf32>
    %14 = stablehlo.subtract %13, %10 : tensor<1x4x256x256xf32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %15 = call @_where(%12, %14, %cst_2) : (tensor<1x4x256x256xi1>, tensor<1x4x256x256xf32>, tensor<f32>) -> tensor<1x4x256x256xf32>
    %16 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x4x256x256xf32>
    %17 = stablehlo.compare  LT, %5, %16,  FLOAT : (tensor<1x4x256x256xf32>, tensor<1x4x256x256xf32>) -> tensor<1x4x256x256xi1>
    %18 = stablehlo.divide %9, %15 : tensor<1x4x256x256xf32>
    %19 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x4x256x256xf32>
    %20 = stablehlo.add %19, %6 : tensor<1x4x256x256xf32>
    %21 = stablehlo.multiply %20, %9 : tensor<1x4x256x256xf32>
    %22 = stablehlo.add %5, %21 : tensor<1x4x256x256xf32>
    %23 = call @_where_0(%17, %18, %22) : (tensor<1x4x256x256xi1>, tensor<1x4x256x256xf32>, tensor<1x4x256x256xf32>) -> tensor<1x4x256x256xf32>
    %24 = stablehlo.multiply %23, %23 : tensor<1x4x256x256xf32>
    %25 = stablehlo.multiply %24, %24 : tensor<1x4x256x256xf32>
    %26 = stablehlo.multiply %25, %25 : tensor<1x4x256x256xf32>
    %27 = stablehlo.reshape %26 : (tensor<1x4x256x256xf32>) -> tensor<4x256x256xf32>
    %28 = stablehlo.dot_general %27, %arg2, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [HIGHEST, HIGHEST] : (tensor<4x256x256xf32>, tensor<1x4x256x64xf32>) -> tensor<4x256x1x64xf32>
    %29 = stablehlo.transpose %28, dims = [2, 0, 1, 3] : (tensor<4x256x1x64xf32>) -> tensor<1x4x256x64xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %30 = stablehlo.reduce(%26 init: %cst_3) applies stablehlo.add across dimensions = [3] : (tensor<1x4x256x256xf32>, tensor<f32>) -> tensor<1x4x256xf32>
    %31 = stablehlo.broadcast_in_dim %30, dims = [0, 1, 2] : (tensor<1x4x256xf32>) -> tensor<1x4x256x1xf32>
    %cst_4 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %32 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x4x256x1xf32>
    %33 = stablehlo.add %31, %32 : tensor<1x4x256x1xf32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0, 1, 2, 3] : (tensor<1x4x256x1xf32>) -> tensor<1x4x256x64xf32>
    %35 = stablehlo.divide %29, %34 : tensor<1x4x256x64xf32>
    return %35 : tensor<1x4x256x64xf32>
  }
  func.func private @_where(%arg0: tensor<1x4x256x256xi1>, %arg1: tensor<1x4x256x256xf32>, %arg2: tensor<f32>) -> tensor<1x4x256x256xf32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<1x4x256x256xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<1x4x256x256xi1>, tensor<1x4x256x256xf32>
    return %2 : tensor<1x4x256x256xf32>
  }
  func.func private @_where_0(%arg0: tensor<1x4x256x256xi1>, %arg1: tensor<1x4x256x256xf32>, %arg2: tensor<1x4x256x256xf32>) -> tensor<1x4x256x256xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1x4x256x256xi1>, tensor<1x4x256x256xf32>
    return %0 : tensor<1x4x256x256xf32>
  }
}
