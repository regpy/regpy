import matplotlib.pyplot as plt

class imshow_fig:
    def __init__(self,nr_rows,nr_cols):
        self.nr_rows = nr_rows
        self.nr_cols = nr_cols
        self.fig, self.ax = plt.subplots(nr_rows, nr_cols, sharex=True, sharey=True)
        self.im = [[[] for i in range(nr_cols)] for j in range(nr_rows)]
        self.cb = [[[] for i in range(nr_cols)] for j in range(nr_rows)]

    def plot(self,plotdata):
        for datum in plotdata:
            row,col = datum['pos']
            assert row <= self.nr_rows
            assert col <= self.nr_cols
            if 'kwargs' in datum:
                self.im[row][col] = self.ax[row,col].imshow(datum['data'],**datum['kwargs'])
            else:
                self.im[row][col] = self.ax[row,col].imshow(datum['data'])
            if 'title' in datum:
                self.ax[row,col].set_title(datum['title'])
            if self.cb[row][col]:
                self.cb[row][col].remove()
            self.cb[row][col]= self.fig.colorbar(self.im[row][col], ax=self.ax[row,col])
        plt.pause(1e-4)