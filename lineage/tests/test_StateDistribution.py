
    def test_MLE_bern(self):
        """ Generate multiple lineages and estimate the bernoulli parameter with MLE. Estimators must be within +/- 0.08 of true locBern for popTime. """
        self.assertTrue(0.899 <= bernoulliParameterEstimatorAnalytical(self.pop1) <= 1.0)

    def test_MLE_exp_analytical(self):
        """ Use the analytical shortcut to estimate the exponential parameters. """
        # test populations w.r.t. time
        beta_out = exponentialAnalytical(self.pop1)
        truther = (45 <= beta_out <= 55)
        self.assertTrue(truther)  # +/- 5 of beta

    def test_MLE_gamma_analytical(self):
        """ Use the analytical shortcut to estimate the Gamma parameters. """
        # test populations w.r.t. time
        #data = sp.gamma.rvs(a = 13, loc = 0 , scale = 3, size = 1000)
        result = gammaAnalytical(self.pop3)
        shape = result[0]
        logging.info('%f : shape estimated.', shape)
        scale = result[1]
        logging.info('%f : scale estimated.', scale)

        self.assertTrue(11 <= shape <= 15)
        self.assertTrue(2 <= scale <= 4)
