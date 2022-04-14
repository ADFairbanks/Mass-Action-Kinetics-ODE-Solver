//Written by Aaron Fairbanks in 2022
//ODE solver that allows for non-stiff solvers to be used specifically for mass action kinetics
/*
To use: 
    Push your kinetic rates into a Rates_Class object. Example: obj.push_rate("a_forward",50);
    Write your ODE function in the form: ODE_Func(state, time, rates_obj)
    Get your kinetic rates from it. Example: rates_obj.get_rate("a_forward")
        (The reason why it's done this way is so it's possible to use the same value multiple times in the ODE function)
    Call Kinetics_ODE_Solve(y0, end_t, func, rates, dt = 0.05, t0 = 0, epsilon = 1e-30)
    Returns the final state. 
      (Changes could be made to return each state, and those are noted in the solver)

Sample usage is provided. 
Currently the solver has a large error with the sample at rates around 50+. Using RK or other methods under the hood should help.
*/
//TODO (future versions):
//  Implement with NumPy
//  Implement other non-stiff methods under the hood (currently uses Euler)
//  Look into adaptive time step

Array.prototype.vectorAdd = function(other) {
    return this.map((x,i) => x + other[i]);
}

Array.prototype.vectorMul = function(f) {
    return this.map(x => f*x);
}

//This class is used for the rate-based ODE solver.
//A class is needed because the ODE should not know implementation details of how its rates are being changed and decoupled.
class Rates_Class{
    constructor(){
        this.unique_rates = {};
        this.initialization_flag = true;
        this.rate_array = [];
        this.current_rate_index = 0;
        this.decoupled_rates = [];
    }
    push_rate(rate_name, rate_value){
        this.unique_rates[rate_name] = rate_value;
    }
    rates_initialization_flag(boolflag){
        this.initialization_flag = boolflag;
        if (this.initialization_flag){
            this.rate_array = [];
            this.current_rate_index = 0;
        }
    }
    get_rate_array(){
        return this.rate_array;
    }
    get_decoupled_rate_array(func){
        this.initialization_flag = false;
        
    }
    set_rate_array(rate_arr){
        this.rate_array = rate_arr;
        this.current_rate_index = 0;
    }
    get_rate(rate_name){
        if (this.initialization_flag){
            //Initialization is used when calling the ODE first to decouple each rate used into separate rates
            this.rate_array.push(this.unique_rates[rate_name]);
            this.current_rate_index += 1;
            return this.unique_rates[rate_name];
        }else{
            //Use the decoupled rates. Note that rate_name isn't used here because the assumption is that the order the rates are accessed is the same.
            var ret = this.rate_array[this.current_rate_index];
            this.current_rate_index += 1;
            return ret;
        }
    }
}

/*
Parameters:
y0: Initial state vector
end_t: Calculate up to this time
func: Function that solves ODE with the parameters: ODE(y_vector, t, rates)
rates: The rates Rates_Class object that 
dt: Given dt to use. Determines accuracy of ODE. 0.05 worked best for what I dealt with.
t0: Start time of ODE. Should be 0 for your first execution.
epsilon: Number slightly higher than 0 that acts as 0 for small calculation errors.
*/
const MAX_ITERATIONS = 10000; //
function Mass_Action_Kinetics_ODE_Solve(y0, end_t, func, rates, dt = 0.05, t0 = 0, epsilon = 1e-30){
    let t=t0; //Current time in ODE
    let ones_y_vector = []; //Used to figure out what rates effect which species
    let rate_usage_vector = []; //2D array of [rates][y_array]. Used = 1, Not_Used = 0
    let individual_rates_vector = []; //Array of rates of all 0's, except for one 1
    let current_rate_adjustment = []; //Rates used are multiplied by this vector. Initialized to an array of 1's.
    let adj_by_y = []; //The rates as indexed by each y. Used to keep track of rate adjustments.
    //initialize vectors
    let sum_rates = []; //The sum of the rates used by each y, to be used when decreasing rates so it doesn't overshoot
    let rate_length = rates.get_rate_array().length;
    for (var i=0; i<y0.length; i++){
        ones_y_vector.push(1.0);
        adj_by_y.push(1.0);
        rate_usage_vector.push([]);
        sum_rates.push(0);
    }
    //Get a new rates array with decoupled rates.
    //Decoupled rates means that rates are duplicated in the array when the ODE uses the same rate multiple times.
    rates.rates_initialization_flag(true);
    func(ones_y_vector, t, rates);
    rates.rates_initialization_flag(false);
    let decoupled_rates = rates.get_rate_array();
    //Initialize a bunch of arrays
    for (var i=0; i<decoupled_rates.length; i++){
        individual_rates_vector.push(0);
        current_rate_adjustment.push(1.0);
    }
    //Call the function with rates individually set to 1 to see which species decrease with the rate
    for (var i=0; i<decoupled_rates.length; i++){
        individual_rates_vector[i] = 1;
        rates.set_rate_array(individual_rates_vector);
        var func_ret = func(ones_y_vector, t, rates);
        rate_sum=0;
        //find which species decrease on this rate
        for (var j=0; j<func_ret.length; j++){
            if (func_ret[j] < 0){
                rate_usage_vector[i].push(1);
                sum_rates[j] += decoupled_rates[i];
            }else{
                rate_usage_vector[i].push(0);
            }
        }
        //set the rate back to call the next one
        individual_rates_vector[i] = 0;
    }
    //Adjust each rate by dt
    let current_rates = decoupled_rates.vectorMul(dt);
    let y_vector = y0;
    //let y_vectors = []; //If you want to return all time steps
    while (t < end_t){
        var exec_count = 0;
        var successful_execution = false;
        //Initialize the adjustment vector to no adjustment (1*rates).
        //We want to keep track of the adjustments but not continually adjust each time step.
        let rate_adj_vector = [];
        for (var i=0; i<decoupled_rates.length; i++){
            rate_adj_vector.push(1.0);
        }
        while (!successful_execution){
            exec_count += 1;
            if (exec_count > MAX_ITERATIONS/dt){
                console.log("ODE went on for too many executions ("+exec_count/dt+")");
                throw "Exiting to too many executions (probably infinite loop of ODE solver)";
            }
            successful_execution = true;
            //Do an execution of the ODE
            rates.set_rate_array(current_rates);
            var func_ret = func(y_vector, t, rates);
            var temp_y = func_ret.vectorAdd(y_vector);
            //Verify the rates are satisfactory
            for (var i=0; i<temp_y.length; i++){
                //One or more rates are too high and results in a negative number, rate(s) need to be adjusted
                if (temp_y[i] < 0){
                    successful_execution = false;
                    //Calculate the result to 0
                    var rate_adj = -temp_y[i] / (y_vector[i] - temp_y[i]);
                    rate_adj = (1 - rate_adj);
                    //Figure out which rates to adjust, and by how much
                    for (var j=0; j<decoupled_rates.length; j++){
                        if (rate_usage_vector[j][i] == 1){
                            //Adjust based on how large each rate is
                            rate_adj_vector[j] = rate_adj * decoupled_rates[j] / sum_rates[i];
                        }
                    }
                    adj_by_y[i] = rate_adj;
                }
                //Rate was adjusted before, and might need to be re-adjusted higher because it is not trying to go into the negatives
                else if (func_ret[i] > epsilon && adj_by_y[i] != 1.0){
                    for (var j=0; j<decoupled_rates.length; j++){
                        if (rate_usage_vector[j][i] == 1){
                            if (current_rates[j] < decoupled_rates[j]*dt){
                                //Adjust rate higher, but not higher than the original rate
                                if (current_rates[j] == 0){
                                    current_rates[j] = Math.min(epsilon,decoupled_rates[j]*dt);
                                }
                                var max_adjust = decoupled_rates[j]*dt/current_rates[j];
                                //I tried a lot of different ways to adjust up. This was the best result. I don't remember how I came up with it despite me writing it an hour ago.
                                var rate_adj = (((func_ret[i]+temp_y[i]))-epsilon)/y_vector[i];
                                rate_adj_vector[j] = Math.min(max_adjust,rate_adj);
                                adj_by_y[i] = rate_adj_vector[j];
                                //successful_execution = false; //If you do this then you likely end up in an infinite loop.
                            }
                        }
                    }
                }else if (adj_by_y[i] > 1){
                    //Reset the increased rate from the previous iteration. This is to avoid ever increasing zig-zagging.
                    for (j=0; j<decoupled_rates.length; j++){
                        if (rate_usage_vector[j][i]==1){
                            var max_adjust = decoupled_rates[j]*dt/current_rates[j];
                            rate_adj_vector[j] /= adj_by_y[i];
                            rate_adj_vector[j] = Math.min(max_adjust,rate_adj_vector[j]);
                            adj_by_y[i] = rate_adj_vector[j];
                        }
                    }
                }
            }
            //Undergo rate adjustment
            for (var j=0; j<rate_adj_vector.length; j++){
                current_rates[j]*=rate_adj_vector[j];
                rate_adj_vector[j]=1;
                //This occasionally gets executed and I forgot why I wrote it. Probably to deal with 1.000000001 stuff.
                //if ((current_rates[j]) > decoupled_rates[j]*dt){
                //    current_rates[j] = decoupled_rates[j]*dt;
                //}
            }
        }
        //Execution successful. Iterate through the ODE with time step dt.
        y_vector = func_ret.vectorAdd(y_vector);
        //y_vectors.push(y_vector); //If you want to return all time steps
        t+=dt;
    }
    return y_vector;
    //return y_vectors; //If you want to return all time steps
}


/*
Sample usage of the following system:
A (k_a)-> B 
B (k_b)<->(k_c_rev) C
C (k_c_fwd)-> D
*/
const SI = {A:0,B:1,C:2,D:3,S_COUNT:4};
function Sample_ODE_Func(y,t,rates_obj){
    //Define each reaction as a separate function
    //A (k_a)-> B 
    function a_b_flow(){
        var y_vector = new Array(SI.S_COUNT).fill(0);
        const k_a = rates_obj.get_rate("k_a");
        const a_flow = y[SI.A]*k_a;
        y_vector[SI.A] = -a_flow;
        y_vector[SI.B] = a_flow;
        return y_vector;
    }
    function b_c_flow(){
        var y_vector = new Array(SI.S_COUNT).fill(0);
        const k_b = rates_obj.get_rate("k_b");
        const b_flow = y[SI.B]*k_b;
        y_vector[SI.B] = -b_flow;
        y_vector[SI.C] = b_flow;
        return y_vector;
    }
    function c_b_flow(){
        var y_vector = new Array(SI.S_COUNT).fill(0);
        const k_c_rev = rates_obj.get_rate("k_c_rev");
        const cb_flow = y[SI.C]*k_c_rev;
        y_vector[SI.B] = cb_flow;
        y_vector[SI.C] = -cb_flow;
        return y_vector;
    }
    function c_d_flow(){
        var y_vector = new Array(SI.S_COUNT).fill(0);
        const k_c_fwd = rates_obj.get_rate("k_c_fwd");
        const cd_flow = y[SI.C]*k_c_fwd;
        y_vector[SI.C] = -cd_flow;
        y_vector[SI.D] = cd_flow;
        return y_vector;
    }
    const reaction_list = [
        a_b_flow,
        b_c_flow,
        c_b_flow,
        c_d_flow
    ]
    var state_vector = new Array(SI.S_COUNT).fill(0);
    for (var i=0; i<reaction_list.length; i++){
        var vector = reaction_list[i]();
        for (var j=0; j<vector.length; j++){
            //Sanity checks
            if (isNaN(vector[j])){
                //Can happen if an initial condition isn't defined or a rate isn't defined or in the rate obj.
                throw "Equation "+i+" in the ODE set on state index "+j+" is not a number";
            }
            if (!isFinite(vector[j])){
                //Can happen if a species isn't subtracted in the model when it usually needs to
                throw "Equation "+i+" in the ODE set on state index "+j+" approaches infinity";
            }
            
            //Add reaction to state
            state_vector[j] += vector[j];
        }
        //state_vector = vector.vectorAdd(state_vector)
    }
    return state_vector;
}

function demo(){
    const param_obj = {};
    param_obj.a = document.getElementById("A_input").value*1.0;
    param_obj.b = document.getElementById("B_input").value*1.0;
    param_obj.c = document.getElementById("C_input").value*1.0;
    param_obj.d = document.getElementById("D_input").value*1.0;
    param_obj.ka = document.getElementById("ka_input").value*1.0;
    param_obj.kb = document.getElementById("kb_input").value*1.0;
    param_obj.kcrev = document.getElementById("kcrev_input").value*1.0;
    param_obj.kcfwd = document.getElementById("kcfwd_input").value*1.0;
    param_obj.time = document.getElementById("time_input").value*1.0;

    const y_vector = call_sample_ode(param_obj);
    const y_labels = ["A","B","C","D"];
    outstr = "";
    for (i=0; i<y_vector.length; i++){
        outstr += y_labels[i]+": "+y_vector[i].toFixed(2)+"<br>";
    }
    document.getElementById("outdiv").innerHTML = outstr;
}

function call_sample_ode(params){
    var rate_obj = new Rates_Class();
    rate_obj.push_rate("k_a",params.ka);
    rate_obj.push_rate("k_b",params.kb);
    rate_obj.push_rate("k_c_rev",params.kcrev);
    rate_obj.push_rate("k_c_fwd",params.kcfwd);
    var y0 = [params.a,params.b,params.c,params.d];
    t = params.time;
    y = Mass_Action_Kinetics_ODE_Solve(y0,t,Sample_ODE_Func,rate_obj,dt=0.05);
    return y;
}
/*

if __name__ == '__main__':
    rate_obj = Rates_Class()
    rate_obj.push_rate("k_a",0.2)
    rate_obj.push_rate("k_b",100)
    rate_obj.push_rate("k_c_rev",50)
    rate_obj.push_rate("k_c_fwd",0.5)
    y0 = [100,0,0,0]
    t = 10
    y = mass_action_ode_solve(y0, t, Sample_ODE_Func, rate_obj)
    y = [round(s,2) for s in y]
    print("Initial state:",y0)
    print("Final state:",y)
*/
