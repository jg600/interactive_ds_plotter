import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.integrate import ode
import sympy as sym

import re
import sys, os

import warnings

def customwarn(message, category, filename, lineno, file=None, line=None):
	open('/dev/null', 'w').write(warnings.formatwarning(message, category, filename, lineno))

warnings.showwarning = customwarn

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

from Tkinter import *
import ttk

root = Tk()

def error_message(opt=None, problem_string=None):
	error_msgs = {
0:'%s is not a valid parameter selection. Entries should be of the form <parameter_name>=<value>' % problem_string
}
	if ((opt != None) & (opt in error_msgs.keys())):
		popup = Toplevel()
		popup.title('Error')
		popup.geometry("200x200")
		msg = Message(popup, text = error_msgs[opt])
		msg.pack()
		ok_button = ttk.Button(popup, text = 'OK', command = popup.destroy)
		ok_button.pack()
	else:
		return None

class TextRedirector(object):
	def __init__(self, widget, tag = "stdout"):
		self.widget = widget
		self.tag = tag

	def write(self, string):
		self.widget.configure(state=NORMAL)
		self.widget.insert(END,string,(self.tag,))
		self.widget.configure(state=DISABLED)

class plotWindow:
	
	def __init__(self, dynsys):
		self.top = Toplevel()
		self.dynsys = dynsys
		self.fig = plt.figure()
		self.ax = plt.axes(xlim = (-10,10), ylim = (-10,10), aspect='equal')
		self.x = []
		self.y = []
		self.line, = self.ax.plot(self.x,self.y)
		self.canvas = FigureCanvasTkAgg(self.fig, master = self.top)
		self.canvas.show()
		self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
		self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.top)
		self.toolbar.update()
		self.canvas._tkcanvas.pack()
		self.info_text = Text(self.top, height = 10)
		self.info_text.pack()
		stdout_holder = sys.stdout
		sys.stdout = TextRedirector(self.info_text, 'stdout')
		print('Jacobian of system:')
		sym.pprint(self.dynsys.Jacobian)
		if self.dynsys.fixedPoints != None and self.dynsys.FPclasses != None:
			for p,c in zip(self.dynsys.fixedPoints, self.dynsys.FPclasses):
				print "Fixed point (%.2f,%.2f) is a %s" % (p[0],p[1],c)
				fp_point, = self.ax.plot(p[0],p[1], 'bx', markersize=6)
				self.fig.canvas.draw()
		sys.stdout = stdout_holder
		

		def draw_soln(event):
			if event.button == 3:
				#print('Clicked at %s,%s' % (event.xdata, event.ydata))
				point, = self.ax.plot(event.xdata, event.ydata, 'ro', markersize=5)
				self.fig.canvas.draw()
				solution = self.dynsys.solveSystem(event.xdata, event.ydata)
				line, = self.ax.plot(solution[0], solution[1], color='blue')
				self.fig.canvas.draw()

		self.canvas.mpl_connect('button_press_event', draw_soln)


class dynSys:
	'Class describing a dynamical system with various attributes'
	
	def __init__(self, maxTime, str1, str2, parameter_list):
		
		self.str1 = str1
		self.str2 = str2
		self.parameter_list = parameter_list
		print(self.parameter_list)
		if self.parameter_list != ['']:
			for p in self.parameter_list:
				p_list = p.split('=')
				param_pattern = re.compile(str(p_list[0]))
				self.str1 = param_pattern.sub(str(p_list[1]), self.str1)
				self.str2 = param_pattern.sub(str(p_list[1]), self.str2)
		self.maxTime = maxTime
		self.t = np.linspace(0, self.maxTime, 1001)
		self.x = sym.symbols('x')
		self.y = sym.symbols('y')
		#self.param_sym_list = [sym.symbols(pname.split('=')[0]) for pname in str(parameter_list.rstrip()).split('\n')]
		self.expr1 = sym.sympify(self.str1)
		self.expr2 = sym.sympify(self.str2)
		self.Jacobian = self.makeJacobian()
		self.fixedPoints = self.getFixedPoints()
		self.FPclasses = self.classifyFPs()
		

	def solveSystem(self, initialx, initialy):
		
		def f(T,Y):
			x = Y[0]
			y = Y[1]
			return [eval(self.str1), eval(self.str2)]

		null_fds = [os.open(os.devnull, os.O_RDWR) for x in xrange(2)]
		save = os.dup(1), os.dup(2)
		os.dup2(null_fds[0],1)
		os.dup2(null_fds[1],2)
		r = ode(f).set_integrator('zvode', method='bdf', with_jacobian=False)
		r.set_initial_value([initialx, initialy], 0)
		X = []
		Y = []
		dt = self.maxTime/10000.0
		while r.successful() and r.t < self.maxTime:
			r.integrate(r.t+dt)
			X.append(r.y[0])
			Y.append(r.y[1])
		os.dup2(save[0],1)
		os.dup2(save[1],2)
		os.close(null_fds[0])
		os.close(null_fds[1])
		return X,Y

	def makeJacobian(self):
		J = sym.Matrix([[self.expr1.diff(self.x), self.expr1.diff(self.y)],[self.expr2.diff(self.x), self.expr2.diff(self.y)]])
		return(J)

	def getFixedPoints(self):
		
		l = []
		
		status = 'neither'
		if ('x' in self.str1) & ('y' in self.str1):
			status = 'both'
			nullclines1 = sym.solve(self.expr1, self.y)
		elif ('x' in self.str1):
			status = 'x'
			nullclines1 = sym.solve(self.expr1, self.x)
		elif ('y' in self.str1):
			status = 'y'
			nullclines1 = sym.solve(self.expr1, self.y)
		else:
			print('No fixed points')
			return None
		
		if ('x' in self.str2) & ('y' in self.str2):
			if (status == 'both'):
				for n in nullclines1:
					expr2_xonly = self.expr2.subs(self.y, n)
					fp_xval = sym.solve(expr2_xonly, self.x)
					for xval in fp_xval:
						l.append([xval.evalf(), n.evalf(subs = {self.x:xval})])
			elif (status == 'x'):
				for n in nullclines1:
					expr2_yonly = self.expr2.subs(self.x, n)
					fp_yvals = sym.solve(expr2_yonly, self.y)
					for yval in fp_yvals:
						l.append([n.evalf(), yval.evalf()])
			elif (status == 'y'):
				for n in nullclines1:
					expr2_xonly = self.expr2.subs(self.y, n)
					fp_xvals = sym.solve(expr2_xonly, self.x)
					for xval in fp_xvals:
						l.append([xval.evalf(), n.evalf()])
						
		elif ('x' in self.str2):
			fp_xvals = sym.solve(self.expr2, self.x)
			if (status == 'both'):
				for xval in fp_xvals:
					for n in nullclines1:
						l.append([xval.evalf(), n.evalf(subs = {self.x:xval})])
			elif (status == 'y'):
				for xval in fp_xvals:
					for n in nullclines1:
						l.append([xval.evalf(), n.evalf()])
			elif (status == 'x'):
				print('Not enough information to find FPs')
				return None
				
		elif ('y' in self.str2):
			fp_yvals = sym.solve(self.expr2, self.y)
			if (status == 'both'):
				for yval in fp_yvals:
					for n in nullclines1:
						xval = sym.solve(n - yval, self.x)
						l.append([xval.evalf(), yval.evalf()])
			elif (status == 'x'):
				for yval in fp_yvals:
					for n in nullclines1:
						l.append([n.evalf(), yval.evalf()])
			elif (status == 'y'):
				print('Not enough information to find FPs')
				return None	
		return(l)

	def classifyFPs(self):
		if self.fixedPoints != None:
			try:
				FPclasses = []
				for f in self.fixedPoints:
					JatFP = self.Jacobian.subs([(self.x,f[0]),(self.y,f[1])])
					detJ = JatFP.det().evalf()
					trJ = JatFP.trace().evalf()
					if detJ < 0:
						FPclasses.append('Saddle')
					elif detJ == 0:
						FPclasses.append('Line of fixed points')
					elif detJ > 0:
						if trJ < 0:
							if detJ < 0.25*trJ**2:
								FPclasses.append('Stable node')
							elif detJ == 0.25*trJ**2:
								FPclasses.append('Improper/Star node')
							elif detJ > 0.25*trJ**2:
								FPclasses.append('Stable focus')
						elif trJ == 0:
							FPclasses.append('Centre')
						elif trJ > 0:
							if detJ < 0.25*trJ**2:
								FPclasses.append('Unstable node')
							elif detJ == 0.25*trJ**2:
								FPclasses.append('Improper/Star node')
							elif detJ > 0.25*trJ**2:
								FPclasses.append('Unstable focus')
				return FPclasses
			except TypeError:
				print('There was a problem with the fixed point classification.')
				return None
			

class initWindow:
	def __init__(self, parent):
		self.mainframe = ttk.Frame(parent, padding="3 3 12 12")
		self.mainframe.grid(column=0, row=0, sticky=(N, W))
		self.mainframe.columnconfigure(0, weight=1)
		self.mainframe.rowconfigure(0, weight=1)

		self.xdot_label = ttk.Label(self.mainframe, width = 7, text = 'dx/dt = ')
		self.xdot_label.grid(column=1, row=1, sticky = (N,W))

		self.xdotStr = StringVar()

		self.xdot_entry = ttk.Entry(self.mainframe, width=10, textvariable=self.xdotStr)
		self.xdot_entry.grid(column=2, row=1, sticky=(N,W))

		self.ydot_label = ttk.Label(self.mainframe, width = 7, text = 'dy/dt = ')
		self.ydot_label.grid(column=1, row=2, sticky = (N,W))

		self.ydotStr = StringVar()

		self.ydot_entry = ttk.Entry(self.mainframe, width=10, textvariable=self.ydotStr)
		self.ydot_entry.grid(column=2, row=2, sticky=(N,W))

		self.maxTime_label = ttk.Label(self.mainframe, width = 7, text = 'Max time = ')
		self.maxTime_label.grid(column=3, row=1, sticky = (N,W))

		self.maxTimeDbl = DoubleVar()

		self.maxTime_entry = ttk.Entry(self.mainframe, width=10, textvariable=self.maxTimeDbl)
		self.maxTime_entry.grid(column=4, row=1, sticky=(N,W))
		self.maxTime_entry.delete(0,END)
		self.maxTime_entry.insert(0,1.0)

		self.parameter_label = Label(self.mainframe, text = "Parameters: ")
		self.parameter_label.grid(column=3, row=2, sticky = (N,W))

		self.parameter_text = Text(self.mainframe, width = 10, height = 5)
		self.parameter_text.grid(column = 4, row = 2, sticky = (N,W))

		self.makeds_button = ttk.Button(self.mainframe, text = 'Visualise', command = self.visualise)
		self.makeds_button.grid(column=3,row=4,sticky = (N,W))

		self.quit_button = ttk.Button(self.mainframe, text = 'Quit', command = self._quit)
		self.quit_button.grid(column=4,row=4,sticky = (N,W))

		for child in self.mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)

	def visualise(self):
		parameter_list = str(self.parameter_text.get('1.0',END).rstrip()).split('\n')
		parameter_pattern = re.compile('^[A-Za-z]{1}\w*={1}\d+(\.{1}\d+)*$')
		for p in parameter_list:
			if p != '' and not re.match(parameter_pattern, p):
				error_message(opt=0, problem_string=p)
				return None
		ds = dynSys(self.maxTimeDbl.get(), self.xdotStr.get(), self.ydotStr.get(), parameter_list)
		plt1 = plotWindow(ds)

	def _quit(self):
		root.quit()
		root.destroy()


init1 = initWindow(root)

root.mainloop()
