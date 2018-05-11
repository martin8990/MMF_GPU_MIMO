
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

class ConvergencePlotRequest : 
    def __init__(self,error : np.array,constellation : np.array,name : str):
        self.error = error
        self.constellation = constellation
        self.name = name

def make_constelation_plot(figure,signal):
        figure.showGrid(x=True, y=True)
        figure.setLabel('left', "Quadrature", units='-')
        figure.setLabel('bottom', "InPhase", units='-')
        figure.setRange(xRange = [-1,1])
        figure.setRange(yRange = [-1,1])

        return figure.plot(signal.real, signal.imag, pen=None, symbol='o', symbolPen=None, symbolSize=4, symbolBrush=(255, 255, 255, 255))
    

class InteractiveConvergencePlot :
    def __init__(self,win, request : ConvergencePlotRequest,lowerbound,upperbound):

        win.nextRow()
        self.errorPlot = win.addPlot(title= request.name + " Error")
        self.errorPlot.plot(request.error, pen=(255,0,0,255))
        self.errorPlot.showGrid(x=True, y=True)

        self.lr = pg.LinearRegionItem([lowerbound,upperbound])
        self.lr.setZValue(-10)
        self.errorPlot.addItem(self.lr)

        

        self.constellation_figure = win.addPlot(title= request.name +" Constellation")
        self.constellation_plot = make_constelation_plot(self.constellation_figure,request.constellation)

        def updateRegion():
            self.lr.setRegion(constellation_figure.getViewBox().viewRange()[0])

        def updatePlot():
            
            r = self.lr.getRegion()
            r_min = int(r[0])
            r_max = int(r[1])

            self.constellation_plot.setData(request.constellation.real[r_min:r_max],request.constellation.imag[r_min:r_max])
            #updateRegion()

        self.lr.sigRegionChanged.connect(updatePlot)
        updatePlot()



def plot_interactive_convergence(convergence_plot_request_list : list,lowerbound,upperbound):


    app = QtGui.QApplication([])
    win = pg.GraphicsWindow(title="Convergence Plotter")
    win.resize(1000,600)
    win.setWindowTitle('Convergence Plotter')
    pg.setConfigOptions(antialias=True)
    
    for request in convergence_plot_request_list:
       convergencePlot = InteractiveConvergencePlot(win,request,lowerbound,upperbound)
    QtGui.QApplication.instance().exec_()
        

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    N = 4 * (10 ** 4)
    x = np.arange(0,N,1)
    debugError = np.sin(x/N)
    
    real = np.random.rand(N)*2-1
    imag = np.random.rand(N)*2-1
    Constellation = real + 1j * imag
    request = [ConvergencePlotRequest(debugError,Constellation,"Test"), ConvergencePlotRequest(debugError,Constellation,"Test2")]
    plot_interactive_convergence(request,30000,31000)

    #if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        
