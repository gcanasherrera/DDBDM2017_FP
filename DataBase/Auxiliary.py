#ALL CREDITS TO: Jarle Brinchmann

def cv1(x, bws, model='gaussian', plot=False, n_folds=10):
    """
        This calculates the leave-one-out cross validation. If you set
        plot to True, then it will show a big grid of the test and training
        samples with the KDE chosen at each step. You might need to modify the
        code if you want a nicer layout :)
        """
    
    # Get the number of bandwidths to check and the number of objects
    N_bw = len(bws)
    N = len(x)
    cv_1 = np.zeros(N_bw)
    
    # If plotting is requested, set up the plot region
    if plot:
        fig, axes = plt.subplots(N_bw, np.ceil(N/n_folds), figsize=(15, 8))
        xplot = np.linspace(-3, 8, 1000)

    # Loop over each band-width and calculate the probability of the
    # test set for this band-width
    for i, bw in enumerate(bws):
        
        # I will do N-fold CV here. This divides X into N_folds
        kf = KFold(N, n_folds=n_folds)
        
        # Initiate - lnP will contain the log likelihood of the test sets
        # and i_k is a counter for the folds that is used for plotting and
        # nothing else..
        lnP = 0.0
        i_k = 0
        
        # Loop over each fold
        for train, test in kf:
            x_train = x[train, :]
            x_test = x[test, :]
            
            # Create the kernel density model for this bandwidth and fit
            # to the training set.
            kde = KernelDensity(kernel=model, bandwidth=bw).fit(x_train)
            
            # score evaluates the log likelihood of a dataset given the fitted KDE.
            log_prob = kde.score(x_test)
            
            if plot:
                # Show the tries
                ax = axes[i][i_k]
                
                # Note that the test sample is hard to see here.
                hist(x_train, bins=10, ax=ax, color='red')
                hist(x_test, bins=10, ax=ax, color='blue')
                ax.plot(xplot, np.exp(kde.score_samples(xplot[:, np.newaxis])))
                i_k += 1
            
            
            lnP += log_prob
        
        # Calculate the average likelihood
        cv_1[i] = lnP/N

    return cv_1
