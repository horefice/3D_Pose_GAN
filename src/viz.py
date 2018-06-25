import numpy as np
import visdom

class Viz(object):
  def __init__(self):
    self.viz = visdom.Visdom(port=8099)
    # self.viz.close(None) #Close all previously

  def create_plot(self, xlabel='', ylabel='', title='', opts_dict={}):
    options = dict(xlabel=xlabel,
      ylabel=ylabel,
      title=title)
    options.update(opts_dict)

    return self.viz.line(X=np.array([0]),
                         Y=np.array([0]),
                         opts=options)

  def update_plot(self, x, y, window, type_upd):
    self.viz.line(X=np.array([x]),
                  Y=np.array([y]),
                  win=window,
                  update=type_upd)

  def create_img(self, x, title=''):
    self.viz.image(x, opts=dict(title=title))

  def create_scatter(self, x, title=''):
    self.viz.scatter(x, opts=dict(title=title))

  def create_hist(self, x, opts={}):
    return self.viz.histogram(np.array([x]), opts=opts)