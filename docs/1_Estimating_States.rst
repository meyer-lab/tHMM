.. code:: ipython3

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline
    import scipy.stats as sp

.. code:: ipython3

    from lineage.LineageTree import LineageTree
    from lineage.states.StateDistributionExponential import StateDistribution
    
    # pi: the initial probability vector
    pi = np.array([0.6, 0.4], dtype="float")
    # Recall that this means that the first cell in our lineage in generation 1 
    # has a 60% change of being state 0 and a 40% chance of being state 1.
    # The values of this vector have to add up to 1 because of the 
    # Law of Total Probability.
    
    # T: transition probability matrix
    T = np.array([[0.75, 0.25],
                  [0.25, 0.75]], dtype="float")
    
    # E: states are defined as StateDistribution objects
    
    # State 0 parameters "Resistant"
    bern_p0 = 1
    gamma_a0 = 7
    
    # State 1 parameters "Susceptible"
    bern_p1 = 1
    gamma_a1 = 49
    
    state_obj0 = StateDistribution(bern_p0, gamma_a0)
    state_obj1 = StateDistribution(bern_p1, gamma_a1)
    
    E = [state_obj0, state_obj1]
    
    lineage1 = LineageTree(pi, T, E, desired_num_cells=2**10 - 1, desired_experiment_time=250, censor_condition=3)
    # These are the minimal arguments required to instantiate lineages

.. code:: ipython3

    a = [cell.obs[1] for cell in lineage1.output_lineage if cell.state==0 and cell.obs[2]==1]
    random_sampler = sp.bernoulli.rvs(p=0.5, size=len(a))
    b = [cell.obs[1] for cell in lineage1.output_lineage if cell.state==0]
    random_samplerb = sp.bernoulli.rvs(p=0.5, size=len(b))
    f = [cell.obs[1] for cell in lineage1.output_lineage if cell.state==0 and cell.obs[2]==0]
    bins=30
    plt.hist(b, alpha=0.5, bins=bins, label='ALL')
    plt.hist(a, alpha=0.5, bins=bins, label='Observed')
    plt.hist(f, alpha=0.5, bins=bins, label='CENSORED')
    plt.legend()
    
    
    print(sum(random_sampler*a)/sum(random_sampler))
    print(sum(random_samplerb*b)/sum(random_samplerb))
    print(sum(f)/len(f))

.. code:: ipython3

    c = [cell.obs[1] for cell in lineage1.output_lineage if cell.state==1 and cell.obs[2]==1]
    random_samplerc = sp.bernoulli.rvs(p=0.5, size=len(c))
    d = [cell.obs[1] for cell in lineage1.output_lineage if cell.state==1]
    random_samplerd = sp.bernoulli.rvs(p=0.5, size=len(d))
    e = [cell.obs[1] for cell in lineage1.output_lineage if cell.state==1 and cell.obs[2]==0]
    plt.hist(d, alpha=0.5, bins=bins, label='ALL')
    plt.hist(c, alpha=0.5, bins=bins, label='Observed')
    plt.hist(e, alpha=0.5, bins=bins, label='CENSORED')
    
    plt.legend()
    
    print(c)
    print(random_samplerc*c)
    print(sum(random_samplerc*c)/sum(random_samplerc))
    print(sum(random_samplerd*d)/sum(random_samplerd))
    print(sum(e)/len(e))

.. code:: ipython3

    def funcer(exp_time):
        # pi: the initial probability vector
        pi = np.array([0.6, 0.4], dtype="float")
        # Recall that this means that the first cell in our lineage in generation 1 
        # has a 60% change of being state 0 and a 40% chance of being state 1.
        # The values of this vector have to add up to 1 because of the 
        # Law of Total Probability.
    
        # T: transition probability matrix
        T = np.array([[0.75, 0.25],
                      [0.25, 0.75]], dtype="float")
    
        # E: states are defined as StateDistribution objects
    
        # State 0 parameters "Resistant"
        bern_p0 = 1
        gamma_a0 = 7
    
        # State 1 parameters "Susceptible"
        bern_p1 = 1
        gamma_a1 = 49
    
        state_obj0 = StateDistribution(bern_p0, gamma_a0)
        state_obj1 = StateDistribution(bern_p1, gamma_a1)
    
        E = [state_obj0, state_obj1]
    
        lineage1 = LineageTree(pi, T, E, desired_num_cells=2**12 - 1, desired_experiment_time=exp_time, censor_condition=3)
        # These are the minimal arguments required to instantiate lineages
        a = [cell.obs[1] for cell in lineage1.output_lineage if cell.state==0 and cell.obs[2]==1]
        random_samplera = np.random.choice(a,size=len(a))
        b = [cell.obs[1] for cell in lineage1.output_lineage if cell.state==0]
        random_samplerb = np.random.choice(b,size=len(b))
        f = [cell.obs[1] for cell in lineage1.output_lineage if cell.state==0 and cell.obs[2]==0]
        c = [cell.obs[1] for cell in lineage1.output_lineage if cell.state==1 and cell.obs[2]==1]
        random_samplerc = np.random.choice(c,size=len(c))
    
        d = [cell.obs[1] for cell in lineage1.output_lineage if cell.state==1]
        random_samplerd = np.random.choice(d,size=len(d))
    
        e = [cell.obs[1] for cell in lineage1.output_lineage if cell.state==1 and cell.obs[2]==0]
        return sum(random_samplera)/len(random_samplera), (sum(random_samplerb)/len(random_samplerb)), (sum(random_samplerc)/len(random_samplerc)), (sum(random_samplerd)/len(random_samplerd)),

.. code:: ipython3

    aa = []
    bb = []
    cc = []
    dd = []
    times = np.linspace(144,500, 100)
    for i in times:
        a,b,c,d = funcer(i)
        aa.append(a)
        bb.append(b)
        cc.append(c)
        dd.append(d)
        

.. code:: ipython3

    plt.scatter(times, aa)
    plt.scatter(times, bb)

.. code:: ipython3

    plt.scatter(times, cc)
    plt.scatter(times, dd)

.. code:: ipython3

    plt.scatter(times, [b-a for a,b in zip(aa,bb)], label='difference between all and observed')
    plt.legend()

.. code:: ipython3

    plt.scatter(times, [7-a for a in aa], label='difference between true and observed')
    plt.scatter(times, [7-a for a in bb], label='difference between true and all')
    plt.hlines(sum([7-a for a in aa])/len([7-a for a in aa]), xmin=min(times),xmax=max(times), color='g')
    plt.hlines(sum([7-a for a in bb])/len([7-a for a in bb]), xmin=min(times),xmax=max(times))
    plt.ylim(0,0.5)
    plt.legend()

.. code:: ipython3

    plt.scatter(times, [b-a for a,b in zip(cc,dd)])

.. code:: ipython3

    plt.scatter(times, [49-a for a in cc], label='difference between true and observed')
    plt.scatter(times, [49-a for a in dd], label='difference between true and all')
    plt.hlines(sum([49-a for a in cc])/len([49-a for a in cc]), xmin=min(times),xmax=max(times), color='g')
    plt.hlines(sum([49-a for a in dd])/len([49-a for a in dd]), xmin=min(times),xmax=max(times))
    plt.legend()


.. code:: ipython3

    print(bern_obs)

