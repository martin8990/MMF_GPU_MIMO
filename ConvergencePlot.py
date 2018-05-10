
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

class ConvergencePlotRequest : 
    def __init__(self,error : np.array,constellation : np.array,name : str):
        self.error = error
        self.constellation = constellation
        self.name = name

class InteractiveConvergencePlot :
    def __init__(self, win, request : ConvergencePlotRequest):

        win.nextRow()
        self.errorPlot = win.addPlot(title= request.name + " Error")
        self.errorPlot.plot(request.error, pen=(255,255,255,200))
        lowerbound = max(0,len(request.error) - 1000 - 10000)
        upperbound = max(0,len(request.error) - 1000)
        self.lr = pg.LinearRegionItem([lowerbound,upperbound])
        self.lr.setZValue(-10)
        self.errorPlot.addItem(self.lr)

        self.constellationPlot = win.addPlot(title= request.name +" Constellation")
        self.constellation = self.constellationPlot.plot(request.constellation.real, request.constellation.imag, pen=None, symbol='t', symbolPen=None, symbolSize=4, symbolBrush=(100, 100, 255, 50))

        def updateRegion():
            self.lr.setRegion(constellationPlot.getViewBox().viewRange()[0])

        def updatePlot():
            
            r = self.lr.getRegion()
            r_min = int(r[0])
            r_max = int(r[1])

            self.constellation.setData(request.constellation.real[r_min:r_max],request.constellation.imag[r_min:r_max])
            #updateRegion()

        self.lr.sigRegionChanged.connect(updatePlot)
        updatePlot()


def plot_interactive_convergence(convergence_plot_request_list : list):
    app = QtGui.QApplication([])

    win = pg.GraphicsWindow(title="Convergence Plotter")
    win.resize(1000,600)
    win.setWindowTitle('Convergence Plotter')
    pg.setConfigOptions(antialias=True)
    
    for request in convergence_plot_request_list:
       convergencePlot = InteractiveConvergencePlot(win,request)
    QtGui.QApplication.instance().exec_()
        

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    x = np.arange(0,100000,1)
    debugError = x*x
    N = len(x)
    real = np.random.rand(N)
    imag = np.random.rand(N)
    Constellation = real + 1j * imag
    request = [ConvergencePlotRequest(debugError,Constellation,"Test"), ConvergencePlotRequest(debugError,Constellation,"Test2")]
    plot_interactive_convergence(request)

    #if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        
