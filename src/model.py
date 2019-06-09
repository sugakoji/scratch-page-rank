import numpy as np

class PageRank():

    def __init__(self, d=0.85, eta=0.000001, max_iter=100):
        self.d = d
        self.eta = eta
        self.mat_iter = max_iter

    def get_rank(self, df):
        url_ls = list(set(
            list(df["url"].values) + list(df["to"].values)))

        url_dic = {_url: i for i, _url in enumerate(url_ls)}

        edges_weight = [
            (url_dic[dd["url"]], url_dic[dd["to"]], dd["weight"]) for
            i, dd in df.iterrows()]

        self.N = len(url_dic.keys())

        mat = np.zeros((self.N, self.N))
        for t in edges_weight:
            mat[t[0], t[1]] = t[2]

        sm = np.sum(mat, axis=1)

        # sthostic
        for i, v in enumerate(sm):
            mat[i] = 1 / self.N if v == 0 else mat[i] / v

        # primitive

        # primitive adjustmentのための行列(G行列を作成)
        pr = np.full_like(np.zeros((self.N, self.N)), 1 / self.N)
        G = self.d * mat + (1 - self.d) * pr
        r = self._cal(G)
        rv_url_dic = {v: k for k, v in url_dic.items()}

        return {rv_url_dic[i]: _rk for i, _rk in enumerate(r)}

    def _cal(self, G):
        # 固有値ベクトルの初期値

        r = np.array([1 / self.N for i in range(self.N)])
        for _ in range(self.mat_iter):
            new_r = np.dot(r.T, G)
            diff = np.sum(np.abs(new_r - r))
            if diff < self.eta:
                break
            r = new_r
        return r
