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
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_slice %arg0, %c, %c, %c, %c, sizes = [1, 4, 128, 64] : (tensor<1x4x256x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x4x128x64xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4x128x64xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x4x128x64xf32>
    %3 = stablehlo.slice %0 [0:1, 0:4, 0:128, 0:1] : (tensor<1x4x128x64xf32>) -> tensor<1x4x128x1xf32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4x128x1xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1x4x128x1xf32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %6:7 = stablehlo.while(%iterArg = %arg1, %iterArg_5 = %arg2, %iterArg_6 = %0, %iterArg_7 = %c_1, %iterArg_8 = %c_0, %iterArg_9 = %2, %iterArg_10 = %5) : tensor<1x4x256x64xf32>, tensor<1x4x256x64xf32>, tensor<1x4x128x64xf32>, tensor<i32>, tensor<i32>, tensor<1x4x128x64xf32>, tensor<1x4x128x1xf32>
     cond {
      %c_11 = stablehlo.constant dense<2> : tensor<i32>
      %23 = stablehlo.compare  LT, %iterArg_7, %c_11,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %23 : tensor<i1>
    } do {
      %23:3 = func.call @None(%iterArg, %iterArg_5, %iterArg_6, %iterArg_8, %iterArg_9, %iterArg_10) : (tensor<1x4x256x64xf32>, tensor<1x4x256x64xf32>, tensor<1x4x128x64xf32>, tensor<i32>, tensor<1x4x128x64xf32>, tensor<1x4x128x1xf32>) -> (tensor<i32>, tensor<1x4x128x64xf32>, tensor<1x4x128x1xf32>)
      %c_11 = stablehlo.constant dense<1> : tensor<i32>
      %24 = stablehlo.add %iterArg_7, %c_11 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_5, %iterArg_6, %24, %23#0, %23#1, %23#2 : tensor<1x4x256x64xf32>, tensor<1x4x256x64xf32>, tensor<1x4x128x64xf32>, tensor<i32>, tensor<i32>, tensor<1x4x128x64xf32>, tensor<1x4x128x1xf32>
    }
    %cst_2 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x4x128x1xf32>
    %8 = stablehlo.add %6#6, %7 : tensor<1x4x128x1xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2, 3] : (tensor<1x4x128x1xf32>) -> tensor<1x4x128x64xf32>
    %10 = stablehlo.divide %6#5, %9 : tensor<1x4x128x64xf32>
    %c_3 = stablehlo.constant dense<128> : tensor<i32>
    %11 = stablehlo.dynamic_slice %arg0, %c, %c, %c_3, %c, sizes = [1, 4, 128, 64] : (tensor<1x4x256x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x4x128x64xf32>
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4x128x64xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<1x4x128x64xf32>
    %14 = stablehlo.slice %11 [0:1, 0:4, 0:128, 0:1] : (tensor<1x4x128x64xf32>) -> tensor<1x4x128x1xf32>
    %15 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4x128x1xf32>
    %16 = stablehlo.multiply %14, %15 : tensor<1x4x128x1xf32>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %17:7 = stablehlo.while(%iterArg = %arg1, %iterArg_5 = %arg2, %iterArg_6 = %11, %iterArg_7 = %c_4, %iterArg_8 = %c_0, %iterArg_9 = %13, %iterArg_10 = %16) : tensor<1x4x256x64xf32>, tensor<1x4x256x64xf32>, tensor<1x4x128x64xf32>, tensor<i32>, tensor<i32>, tensor<1x4x128x64xf32>, tensor<1x4x128x1xf32>
     cond {
      %c_11 = stablehlo.constant dense<2> : tensor<i32>
      %23 = stablehlo.compare  LT, %iterArg_7, %c_11,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %23 : tensor<i1>
    } do {
      %23:3 = func.call @None_1(%iterArg, %iterArg_5, %iterArg_6, %iterArg_8, %iterArg_9, %iterArg_10) : (tensor<1x4x256x64xf32>, tensor<1x4x256x64xf32>, tensor<1x4x128x64xf32>, tensor<i32>, tensor<1x4x128x64xf32>, tensor<1x4x128x1xf32>) -> (tensor<i32>, tensor<1x4x128x64xf32>, tensor<1x4x128x1xf32>)
      %c_11 = stablehlo.constant dense<1> : tensor<i32>
      %24 = stablehlo.add %iterArg_7, %c_11 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_5, %iterArg_6, %24, %23#0, %23#1, %23#2 : tensor<1x4x256x64xf32>, tensor<1x4x256x64xf32>, tensor<1x4x128x64xf32>, tensor<i32>, tensor<i32>, tensor<1x4x128x64xf32>, tensor<1x4x128x1xf32>
    }
    %18 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x4x128x1xf32>
    %19 = stablehlo.add %17#6, %18 : tensor<1x4x128x1xf32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1, 2, 3] : (tensor<1x4x128x1xf32>) -> tensor<1x4x128x64xf32>
    %21 = stablehlo.divide %17#5, %20 : tensor<1x4x128x64xf32>
    %22 = stablehlo.concatenate %10, %21, dim = 2 : (tensor<1x4x128x64xf32>, tensor<1x4x128x64xf32>) -> tensor<1x4x256x64xf32>
    return %22 : tensor<1x4x256x64xf32>
  }
  func.func private @None(%arg0: tensor<1x4x256x64xf32>, %arg1: tensor<1x4x256x64xf32>, %arg2: tensor<1x4x128x64xf32>, %arg3: tensor<i32>, %arg4: tensor<1x4x128x64xf32>, %arg5: tensor<1x4x128x1xf32>) -> (tensor<i32>, tensor<1x4x128x64xf32>, tensor<1x4x128x1xf32>) {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.add %arg3, %c : tensor<i32>
    %c_0 = stablehlo.constant dense<128> : tensor<i32>
    %1 = stablehlo.multiply %arg3, %c_0 : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %2 = stablehlo.compare  LT, %1, %c_1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = stablehlo.convert %1 : tensor<i32>
    %c_2 = stablehlo.constant dense<256> : tensor<i32>
    %4 = stablehlo.add %3, %c_2 : tensor<i32>
    %5 = stablehlo.select %2, %4, %1 : tensor<i1>, tensor<i32>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.dynamic_slice %arg0, %c_3, %c_3, %5, %c_3, sizes = [1, 4, 128, 64] : (tensor<1x4x256x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x4x128x64xf32>
    %7 = stablehlo.multiply %arg3, %c_0 : tensor<i32>
    %8 = stablehlo.compare  LT, %7, %c_1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %9 = stablehlo.convert %7 : tensor<i32>
    %10 = stablehlo.add %9, %c_2 : tensor<i32>
    %11 = stablehlo.select %8, %10, %7 : tensor<i1>, tensor<i32>
    %12 = stablehlo.dynamic_slice %arg1, %c_3, %c_3, %11, %c_3, sizes = [1, 4, 128, 64] : (tensor<1x4x256x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x4x128x64xf32>
    %13 = stablehlo.transpose %6, dims = [0, 1, 3, 2] : (tensor<1x4x128x64xf32>) -> tensor<1x4x64x128xf32>
    %14 = stablehlo.reshape %arg2 : (tensor<1x4x128x64xf32>) -> tensor<4x128x64xf32>
    %15 = stablehlo.dot_general %14, %13, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x128x64xf32>, tensor<1x4x64x128xf32>) -> tensor<4x128x1x128xf32>
    %16 = stablehlo.transpose %15, dims = [2, 0, 1, 3] : (tensor<4x128x1x128xf32>) -> tensor<1x4x128x128xf32>
    %cst = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4x128x128xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<1x4x128x128xf32>
    %19 = stablehlo.multiply %18, %18 : tensor<1x4x128x128xf32>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x4x128x128xf32>
    %21 = stablehlo.add %20, %19 : tensor<1x4x128x128xf32>
    %22 = stablehlo.rsqrt %21 : tensor<1x4x128x128xf32>
    %23 = stablehlo.multiply %18, %22 : tensor<1x4x128x128xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x4x128x128xf32>
    %25 = stablehlo.compare  LT, %18, %24,  FLOAT : (tensor<1x4x128x128xf32>, tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xi1>
    %26 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x4x128x128xf32>
    %27 = stablehlo.subtract %26, %23 : tensor<1x4x128x128xf32>
    %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %28 = call @_where(%25, %27, %cst_6) : (tensor<1x4x128x128xi1>, tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128x128xf32>
    %29 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x4x128x128xf32>
    %30 = stablehlo.compare  LT, %18, %29,  FLOAT : (tensor<1x4x128x128xf32>, tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xi1>
    %31 = stablehlo.divide %22, %28 : tensor<1x4x128x128xf32>
    %32 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x4x128x128xf32>
    %33 = stablehlo.add %32, %19 : tensor<1x4x128x128xf32>
    %34 = stablehlo.multiply %33, %22 : tensor<1x4x128x128xf32>
    %35 = stablehlo.add %18, %34 : tensor<1x4x128x128xf32>
    %36 = call @_where_0(%30, %31, %35) : (tensor<1x4x128x128xi1>, tensor<1x4x128x128xf32>, tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xf32>
    %37 = stablehlo.multiply %36, %36 : tensor<1x4x128x128xf32>
    %38 = stablehlo.multiply %37, %37 : tensor<1x4x128x128xf32>
    %39 = stablehlo.multiply %38, %38 : tensor<1x4x128x128xf32>
    %40 = stablehlo.reshape %39 : (tensor<1x4x128x128xf32>) -> tensor<4x128x128xf32>
    %41 = stablehlo.dot_general %40, %12, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf32>, tensor<1x4x128x64xf32>) -> tensor<4x128x1x64xf32>
    %42 = stablehlo.transpose %41, dims = [2, 0, 1, 3] : (tensor<4x128x1x64xf32>) -> tensor<1x4x128x64xf32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %43 = stablehlo.reduce(%39 init: %cst_7) applies stablehlo.add across dimensions = [3] : (tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %44 = stablehlo.broadcast_in_dim %43, dims = [0, 1, 2] : (tensor<1x4x128xf32>) -> tensor<1x4x128x1xf32>
    %45 = stablehlo.add %arg4, %42 : tensor<1x4x128x64xf32>
    %46 = stablehlo.add %arg5, %44 : tensor<1x4x128x1xf32>
    return %0, %45, %46 : tensor<i32>, tensor<1x4x128x64xf32>, tensor<1x4x128x1xf32>
  }
  func.func private @_where(%arg0: tensor<1x4x128x128xi1>, %arg1: tensor<1x4x128x128xf32>, %arg2: tensor<f32>) -> tensor<1x4x128x128xf32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<1x4x128x128xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<1x4x128x128xi1>, tensor<1x4x128x128xf32>
    return %2 : tensor<1x4x128x128xf32>
  }
  func.func private @_where_0(%arg0: tensor<1x4x128x128xi1>, %arg1: tensor<1x4x128x128xf32>, %arg2: tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1x4x128x128xi1>, tensor<1x4x128x128xf32>
    return %0 : tensor<1x4x128x128xf32>
  }
  func.func private @None_1(%arg0: tensor<1x4x256x64xf32>, %arg1: tensor<1x4x256x64xf32>, %arg2: tensor<1x4x128x64xf32>, %arg3: tensor<i32>, %arg4: tensor<1x4x128x64xf32>, %arg5: tensor<1x4x128x1xf32>) -> (tensor<i32>, tensor<1x4x128x64xf32>, tensor<1x4x128x1xf32>) {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.add %arg3, %c : tensor<i32>
    %c_0 = stablehlo.constant dense<128> : tensor<i32>
    %1 = stablehlo.multiply %arg3, %c_0 : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %2 = stablehlo.compare  LT, %1, %c_1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = stablehlo.convert %1 : tensor<i32>
    %c_2 = stablehlo.constant dense<256> : tensor<i32>
    %4 = stablehlo.add %3, %c_2 : tensor<i32>
    %5 = stablehlo.select %2, %4, %1 : tensor<i1>, tensor<i32>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.dynamic_slice %arg0, %c_3, %c_3, %5, %c_3, sizes = [1, 4, 128, 64] : (tensor<1x4x256x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x4x128x64xf32>
    %7 = stablehlo.multiply %arg3, %c_0 : tensor<i32>
    %8 = stablehlo.compare  LT, %7, %c_1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %9 = stablehlo.convert %7 : tensor<i32>
    %10 = stablehlo.add %9, %c_2 : tensor<i32>
    %11 = stablehlo.select %8, %10, %7 : tensor<i1>, tensor<i32>
    %12 = stablehlo.dynamic_slice %arg1, %c_3, %c_3, %11, %c_3, sizes = [1, 4, 128, 64] : (tensor<1x4x256x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x4x128x64xf32>
    %13 = stablehlo.transpose %6, dims = [0, 1, 3, 2] : (tensor<1x4x128x64xf32>) -> tensor<1x4x64x128xf32>
    %14 = stablehlo.reshape %arg2 : (tensor<1x4x128x64xf32>) -> tensor<4x128x64xf32>
    %15 = stablehlo.dot_general %14, %13, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x128x64xf32>, tensor<1x4x64x128xf32>) -> tensor<4x128x1x128xf32>
    %16 = stablehlo.transpose %15, dims = [2, 0, 1, 3] : (tensor<4x128x1x128xf32>) -> tensor<1x4x128x128xf32>
    %cst = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4x128x128xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<1x4x128x128xf32>
    %19 = stablehlo.multiply %18, %18 : tensor<1x4x128x128xf32>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x4x128x128xf32>
    %21 = stablehlo.add %20, %19 : tensor<1x4x128x128xf32>
    %22 = stablehlo.rsqrt %21 : tensor<1x4x128x128xf32>
    %23 = stablehlo.multiply %18, %22 : tensor<1x4x128x128xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x4x128x128xf32>
    %25 = stablehlo.compare  LT, %18, %24,  FLOAT : (tensor<1x4x128x128xf32>, tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xi1>
    %26 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x4x128x128xf32>
    %27 = stablehlo.subtract %26, %23 : tensor<1x4x128x128xf32>
    %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %28 = call @_where(%25, %27, %cst_6) : (tensor<1x4x128x128xi1>, tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128x128xf32>
    %29 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x4x128x128xf32>
    %30 = stablehlo.compare  LT, %18, %29,  FLOAT : (tensor<1x4x128x128xf32>, tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xi1>
    %31 = stablehlo.divide %22, %28 : tensor<1x4x128x128xf32>
    %32 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x4x128x128xf32>
    %33 = stablehlo.add %32, %19 : tensor<1x4x128x128xf32>
    %34 = stablehlo.multiply %33, %22 : tensor<1x4x128x128xf32>
    %35 = stablehlo.add %18, %34 : tensor<1x4x128x128xf32>
    %36 = call @_where_0(%30, %31, %35) : (tensor<1x4x128x128xi1>, tensor<1x4x128x128xf32>, tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xf32>
    %37 = stablehlo.multiply %36, %36 : tensor<1x4x128x128xf32>
    %38 = stablehlo.multiply %37, %37 : tensor<1x4x128x128xf32>
    %39 = stablehlo.multiply %38, %38 : tensor<1x4x128x128xf32>
    %40 = stablehlo.reshape %39 : (tensor<1x4x128x128xf32>) -> tensor<4x128x128xf32>
    %41 = stablehlo.dot_general %40, %12, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf32>, tensor<1x4x128x64xf32>) -> tensor<4x128x1x64xf32>
    %42 = stablehlo.transpose %41, dims = [2, 0, 1, 3] : (tensor<4x128x1x64xf32>) -> tensor<1x4x128x64xf32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %43 = stablehlo.reduce(%39 init: %cst_7) applies stablehlo.add across dimensions = [3] : (tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %44 = stablehlo.broadcast_in_dim %43, dims = [0, 1, 2] : (tensor<1x4x128xf32>) -> tensor<1x4x128x1xf32>
    %45 = stablehlo.add %arg4, %42 : tensor<1x4x128x64xf32>
    %46 = stablehlo.add %arg5, %44 : tensor<1x4x128x1xf32>
    return %0, %45, %46 : tensor<i32>, tensor<1x4x128x64xf32>, tensor<1x4x128x1xf32>
  }
}
