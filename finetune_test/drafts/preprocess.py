import os

def namelist(path, fltr):
    cip = []
    for root, structure, files in os.walk(path):
        for file in files:
            if fltr in file:
                cip.append(os.path.join(root, file))
    return cip

with h5py.File('C:/circles/finetune_test/csv/canonical_view_moments_598.h5', 'w') as f:
    f.create_dataset('dataset', data=car)



car = np.zeros([len(cip), 6])
for i, k in enumerate(cip):
    image = np.asarray(open3d.io.read_point_cloud(k).points)
    # image = np.asarray(open3d.io.read_point_cloud('C:/cherepashkin1/phenoseed/598/1484717/1491988_Surface.ply').points)
    X = image[:, 0]
    Y = image[:, 1]
    Z = image[:, 2]
    car[i, :] = np.array([np.matmul(X, X.T), np.matmul(X, Y.T), np.matmul(X, Z.T),
                          np.matmul(Y, Y.T), np.matmul(Y, Z.T), np.matmul(Z, Z.T)])

#     C = np.zeros([3, 3])
#     C[0, 0] = np.matmul(X, X.T)
#     C[0, 1] = np.matmul(X, Y.T)
#     C[0, 2] = np.matmul(X, Z.T)
#     C[1, 0] = C[0, 1]
#     C[1, 1] = np.matmul(Y, Y.T)
#     C[1, 2] = np.matmul(Y, Z.T)
#     C[2, 0] = C[0, 2]
#     C[2, 1] = C[1, 2]
#     C[2, 2] = np.matmul(Z, Z.T)

C2 = np.array([[car[0,0], car[0,1], car[0,2]],
      [car[0,1], car[0,3], car[0,4]],
      [car[0,2], car[0,4], car[0,5]]])

moments = lframe[['moment' + str(i) for i in range(6)]].to_numpy()

# Get orientation matrix from surface pcd with subtracted mean
cip = namelist('C:/cherepashkin1/phenoseed/598', 'Surface.ply')
car2 = np.zeros([len(cip), 3, 3])
car = np.zeros(6)
for i, k in enumerate(cip):
    pcd = np.asarray(open3d.io.read_point_cloud(k).points)
    pcd = pcd - np.mean(pcd, axis=0)
    X = pcd[:, 0]
    Y = pcd[:, 1]
    Z = pcd[:, 2]
    car = np.array([np.matmul(X, X.T), np.matmul(X, Y.T), np.matmul(X, Z.T),
                    np.matmul(Y, Y.T), np.matmul(Y, Z.T), np.matmul(Z, Z.T)])
    _, car2[i, :, :] = LA.eig(pose6tomat(car))



