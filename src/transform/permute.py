import torch


def shuffle_backwards(shuffle):
    shuffle = torch.as_tensor(shuffle)
    a = torch.arange(len(shuffle))
    a[shuffle] = torch.arange(len(shuffle))
    return a


class PermuteNodes:
    def __init__(self, shuffle=None):
        """
        Set shuffle = None to randomize for every __call__
        """
        self.shuffle = shuffle
        if self.shuffle is not None:
            self.backward_shuffle = shuffle_backwards(shuffle)
        else:
            self.backward_shuffle = None

    def __call__(self, data):
        if self.shuffle is None:
            shuffle = torch.randperm(data.pos.shape[0])
            backward_shuffle = shuffle_backwards(shuffle)
        else:
            shuffle = torch.as_tensor(self.shuffle)
            backward_shuffle = torch.as_tensor(self.backward_shuffle)

        data_shuffle = data.clone()

        node_numbers = torch.arange(len(data.pos))  # target node numbers
        node_numbers_shuffle = node_numbers.clone()[shuffle]

        # shuffle sends shuffle to torch.arange(n) indices, so
        # pos[shuffle[0]] = pos_shuffle[shuffle[0]]
        pos = data.pos
        pos_shuffle = pos.clone()[shuffle]
        data_shuffle.pos = pos_shuffle

        face = data.face
        face_indices_shuffle = node_numbers.clone()[backward_shuffle]
        face_shuffle = face.clone()
        face_shuffle[0], face_shuffle[1], face_shuffle[2] = (
            face_indices_shuffle.clone()[face_shuffle[0]],
            face_indices_shuffle.clone()[face_shuffle[1]],
            face_indices_shuffle.clone()[face_shuffle[2]],
        )
        data_shuffle.face = face_shuffle

        if data.get("y") is not None:
            # data_shuffle.y = data.y.clone()[shuffle]
            data_shuffle.y = node_numbers_shuffle

        if data.get("normal") is not None:
            data_shuffle.normal = data.normal.clone()[shuffle]

        if data.get("edge_index") is not None:
            edge_indices_shuffle = node_numbers.clone()[backward_shuffle]
            edge_index_shuffle = data.edge_index.clone()
            edge_index_shuffle[0], edge_index_shuffle[1] = (
                edge_indices_shuffle.clone()[edge_index_shuffle[0]],
                edge_indices_shuffle.clone()[edge_index_shuffle[1]],
            )
            data_shuffle.edge_index = edge_index_shuffle

        return data_shuffle
