#Written by Aaron Fairbanks in 2022
#ODE solver that allows for non-stiff solvers to be used specifically for mass action kinetics
'''
To use: 
  Push your kinetic rates into a Rates_Class object. Example: obj.push_rate("a_forward",50);
  Write your ODE function in the form: ODE_Func(state, time, rates_obj)
  Get your kinetic rates from it. Example: rates_obj.get_rate("a_forward")
      (The reason why it's done this way is so it's possible to use the same value multiple times in the ODE function)
  Call Kinetics_ODE_Solve(y0, end_t, func, rates, dt = 0.05, t0 = 0, epsilon = 1e-30)
  Returns the final state. 
      (Changes could be made to return each state, and those are noted in the solver)

Sample usage is provided
Currently the solver has a large error with the sample at rates around 50+. Using RK or other methods under the hood should help.
'''
#TODO (future versions):
#   Implement with NumPy
#   Implement other non-stiff methods under the hood (currently uses Euler)
#   Look into adaptive time step
from enum import IntEnum

def arrayMul(base_arr, scalar_mult):
    return [x*scalar_mult for x in base_arr]
def arrayAdd(base_arr, add_arr):
    ret = []
    for i in range(len(base_arr)):
        ret.append(base_arr[i]+add_arr[i])
    return ret

class Rates_Class:
    def __init__(self):
        self.unique_rates = {}
        self.initialization_flag = True
        self.rate_array = []
        self.current_rate_index = 0
    def push_rate(self, rate_name, rate_value):
        self.unique_rates[rate_name] = rate_value
    def rates_initialization_flag(self, boolflag):
        self.initialization_flag = boolflag
        if self.initialization_flag:
            self.rate_array = []
            self.current_rate_index = 0
    def get_rate_array(self):
        return self.rate_array
    def set_rate_array(self, rate_arr):
        self.rate_array = rate_arr
        self.current_rate_index = 0
    def get_rate(self, rate_name):
        if self.initialization_flag:
            #Initialization is used when calling the ODE function first to decouple each rate used into separate rates
            self.rate_array.append(self.unique_rates[rate_name])
            self.current_rate_index += 1
            return self.unique_rates[rate_name]
        else:
            #Use the decoupled rates. Note that rate_name isn't used here because the presumption is that the order the rates are accessed is the same
            ret = self.rate_array[self.current_rate_index]
            self.current_rate_index += 1
            return ret
    def __len__(self):
        return len(self.rate_array)
        
MAX_ITERATIONS = 10000
def mass_action_ode_solve(y0, end_t, func, rates, dt = 0.05, t0 = 0, epsilon = 1e-30):
    t = t0
    ones_y_vector = [1.0]*len(y0) #Used to figure out what rates effect which species
    adj_by_y = [1.0]*len(y0) #The rates as indexed by each y. Used to keep track of rate adjustments
    sum_rates = [0]*len(y0)
    
    #Get a new rates array with decoupled rates
    #Decoupled rates means that rates are duplicated in the array when the ODE uses the same rate name multiple times
    rates.rates_initialization_flag(True)
    func(ones_y_vector, t0, rates)
    rates.rates_initialization_flag(False)
    decoupled_rates = rates.get_rate_array()
    
    #Needs to be defined now because len(rates) refers to the length of the decoupled array
    individual_rates_vector = [0]*len(rates)  #Array of all rates of all 0's, except for on 1
    current_rate_adjustment = [1.0]*len(rates) #Rates used are multiplied by this vector
    rate_usage_vector = [[0]*len(y0) for x in range(len(rates))] #2D array of [rates][y_array]. Used = 1, Not Used = 0
    
    for i in range(len(rates)):
        individual_rates_vector[i] = 1
        rates.set_rate_array(individual_rates_vector)
        func_ret = func(ones_y_vector, t0, rates)
        #Find which species decrease on this rate
        for j in range(len(func_ret)):
            if func_ret[j] < 0:
                rate_usage_vector[i][j] = 1
                sum_rates[j] += decoupled_rates[i]
            else:
                rate_usage_vector[i][j] = 0
        individual_rates_vector[i] = 0
    #Adjust rates by each dt
    current_rates = arrayMul(decoupled_rates,dt)
    y_vector = y0
    #y_vectors = [] #for storing time steps
    while t < end_t:
        exec_count = 0
        successful_execution = False
        #Initialized to no adjustment
        rate_adj_vector = [1.0]*len(decoupled_rates)
        while not successful_execution:
            exec_count += 1
            if exec_count > MAX_ITERATIONS/dt:
                raise RuntimeError("Exiting to too many executions in ODE solver") 
            successful_execution = True
            #Execution of ODE
            rates.set_rate_array(current_rates)
            func_ret = func(y_vector, t, rates)
            temp_y = arrayAdd(func_ret, y_vector)
            for i in range(len(temp_y)):
                #rate too high since it causes a negative number
                if temp_y[i] < 0: 
                    successful_execution = False
                    #Calculate the rate needed to make the result 0
                    rate_adj =  -temp_y[i] / (y_vector[i] - temp_y[i])
                    rate_adj = (1 - rate_adj)
                    #Spread those adjustments 
                    for j in range(len(decoupled_rates)):
                        if rate_usage_vector[j][i] == 1:
                            #Adjust based on how large each rate is
                            rate_adj_vector[j] = rate_adj * decoupled_rates[j] / sum_rates[i];
                    adj_by_y[i] = rate_adj
                #rate was adjusted down
                elif temp_y[i] > epsilon and adj_by_y[i] != 1.0:
                    for j in range(len(decoupled_rates)):
                        if rate_usage_vector[j][i] == 1:
                            if current_rates[j] < decoupled_rates[j]*dt:
                                #Adjust rate higher, but not higher than the original rate
                                if current_rates[j] == 0:
                                    current_rates[j] = min(epsilon,decoupled_rates[j]*dt)
                                max_adjust = decoupled_rates[j]*dt / current_rates[j]
                                #I tried a lot of different ways to adjust up. This was the best result. I don't remember how I came up with it despite me writing it an hour ago.
                                rate_adj = ((func_ret[i]+temp_y[i]) - epsilon)/y_vector[i]
                                rate_adj_vector[j]= min(max_adjust,rate_adj)
                                adj_by_y[i] = rate_adj_vector[j]
                                #successful_execution=False #Do not do this. High likelyhood of infinite loop.
                elif adj_by_y[i] > 1:
                    #Reset the increased rate from the previous iteration. This is to avoid ever increasing zig-zaggin
                    for j in range(len(decoupled_rates)):
                        if rate_usage_vector[j][i]==1:
                            max_adjust = decoupled_rates[j]*dt / current_rates[j]
                            rate_adj_vector[j] /= adj_by_y[i]
                            rate_adj_vector[j] = min(max_adjust, rate_adj_vector[j])
                            adj_by_y[i] = rate_adj_vector[j]
            #Undergo rate adjustment
            for j in range(len(rate_adj_vector)):
                current_rates[j]*=rate_adj_vector[j]
                rate_adj_vector[j]=1
                if (current_rates[j]/dt) > decoupled_rates[j]:
                    current_rates[j] = decoupled_rates[j]*dt
        #Execution successful.
        y_vector = arrayAdd(func_ret, y_vector) #Euler's method
        #y_vectors.append(y_vector) #If you want all the time steps
        t+=dt
    return y_vector
    #return y_vectors

#Sample usage of the following system:
#A (k_a)-> B 
#B (k_b)<->(k_c_rev) C
#C (k_c_fwd)-> D
class SI(IntEnum): #shortform for SPECIES_INDEX
    A=0
    B=1
    C=2
    D=3
def Sample_ODE_Func(y,t,rates_obj):
    #Define each reaction as a separate function
    #A (k_a)-> B 
    def a_b_flow():
        y_vector = [0]*len(y)
        k_a = rates_obj.get_rate("k_a")
        a_flow = y[SI.A] * k_a
        y_vector[SI.A] = -a_flow
        y_vector[SI.B] = a_flow
        return y_vector
    #B (k_b)<->(k_c_rev) C.  B->C portion
    def b_c_flow():
        y_vector = [0]*len(y)
        k_b = rates_obj.get_rate("k_b")
        b_flow = y[SI.B] * k_b
        y_vector[SI.B] = -b_flow
        y_vector[SI.C] = b_flow
        return y_vector
    #B (k_b)<->(k_c_rev) C.  B<-C portion
    def c_b_flow():
        y_vector = [0]*len(y)
        k_c_rev = rates_obj.get_rate("k_c_rev")
        cb_flow = y[SI.C] * k_c_rev
        y_vector[SI.B] = cb_flow
        y_vector[SI.C] = -cb_flow
        return y_vector
    #C (k_c_fwd)-> D7
    def c_d_flow():
        y_vector = [0]*len(y)
        k_c_fwd = rates_obj.get_rate("k_c_fwd")
        cd_flow = y[SI.C] * k_c_fwd
        y_vector[SI.C] = -cd_flow
        y_vector[SI.D] = cd_flow
        return y_vector
    #Reaction list
    reactions = [
        a_b_flow,
        b_c_flow,
        c_b_flow,
        c_d_flow
    ]
    #Combine the effect of each reaction
    y_full_vector = [0]*len(y)
    for reaction_func in reactions:
        y_vector_reaction = reaction_func()
        y_full_vector = arrayAdd(y_full_vector,y_vector_reaction)
    return y_full_vector

def Sample_ODE_Short(t,y0):
    y = [0]*len(y0)
    y[0] = -y0[0]*ka
    y[1] = y0[0]*ka + y0[2]*kcrev - y0[1]*kb 
    y[2] = y0[1]*kb - y0[2]*kcrev - y0[2]*kcfwd
    y[3] = y0[2]*kcfwd
    return y

if __name__ == '__main__':
    ka = 0.2
    kb = 10
    kcrev = 5
    kcfwd = 0.5
    t=10
    y0 = [100,0,0,0]
    t = 10
    rate_obj = Rates_Class()
    rate_obj.push_rate("k_a",ka)
    rate_obj.push_rate("k_b",kb)
    rate_obj.push_rate("k_c_rev",kcrev)
    rate_obj.push_rate("k_c_fwd",kcfwd)
    y = mass_action_ode_solve(y0, t, Sample_ODE_Func, rate_obj,dt=0.05)
    y = [round(s,2) for s in y]
    print("Initial state:",y0)
    print("Final state:",y)
    #Scipy's default solver for comparison
    from scipy import integrate
    y_odeint = integrate.solve_ivp(Sample_ODE_Short,(0,t),y0)
    y_o = [round(x[-1],2) for x in y_odeint.y]
    print ("Scipy ivp state:",y_o)