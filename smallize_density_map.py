def smallize_density_map(density_map, stride=1):
    density_map_stride = density_map
    if stride > 1:
        density_map_stride = np.zeros((np.asarray(density_map.shape).astype(int)//stride).tolist(), dtype=np.float32)
        for r in range(density_map_stride.shape[0]):
            for c in range(density_map_stride.shape[1]):
                density_map_stride[r, c] = np.sum(density_map[r*stride:(r+1)*stride, c*stride:(c+1)*stride])
                
    return density_map_stride