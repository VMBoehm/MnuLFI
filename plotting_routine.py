from getdist import MCSamples, plots
import matplotlib as mpl
import matplotlib.pyplot as plt
show_plot=True
  
def adv_triangle_plot(prior, val_true, val_fid=None, samples = None, savefig = False, filename = None, tru_c=None, fid_c=None, prior_cc=None, alphas=[None,None], plot_priors=False, plot_fid=True):

      # Set samples to the posterior samples by default
      if samples is None:
          samples = [self.posterior_samples]
      mc_samples = [MCSamples(samples=s, names = DelfiEnsemble.names, labels = DelfiEnsemble.labels, ranges = DelfiEnsemble.ranges) for s in samples]

      if None in alphas:
          alphas =[0.3,0.7]
      if fid_c == None:
          fid_c = 'orange'
      if tru_c == None:
          tru_c = 'crimson'
      if prior_cc == None:
          prior_cc = 'gold'
      pp = prior
      # Triangle plot
      with mpl.rc_context():
          g = plots.getSubplotPlotter(width_inch = 12)
          g.settings.figure_legend_frame = False
          g.settings.alpha_filled_add=0.6
          g.settings.axes_fontsize=14
          g.settings.legend_fontsize=16
          g.settings.lab_fontsize=20
          g.triangle_plot(mc_samples, filled_compare=True, normalized=True)
          for i in range(0, len(samples[0][0,:])):
              for j in range(0, i+1):
                  ax = g.subplots[i,j]

                  if i is not j:
                      if plot_priors:
                        if isinstance(pp,priors.TruncatedGaussian):

                            x = np.linspace(pp.lower[i],pp.upper[i],100)
                            y = np.linspace(pp.lower[j],pp.upper[j],100)
                            X, Y = np.meshgrid(x,y)
                            pos = np.empty(X.shape + (2,))
                            pos[:, :, 0] = X; pos[:, :, 1] = Y
                            var = multivariate_normal(mean=[pp.mean[i],pp.mean[j]], cov=[[pp.C[i,i],pp.C[i,j]],[pp.C[j,i],pp.C[j,j]]])
                            Z   = var.pdf(pos)
                            norm = multivariate_normal.pdf([0.,0.],mean=[0.,0.], cov=[[pp.C[i,i],pp.C[i,j]],[pp.C[j,i],pp.C[j,j]]])
                            Z = -2*np.log(Z/norm)   # returns the chi squared

                            # # confidence level contours
                            conf_level = np.array([0.68, 0.95])
                            chi2 = -2. * np.log(1. - conf_level)

                            ax.contourf(X,Y,Z, [0., chi2[0]], colors=prior_cc, alpha=alphas[1],zorder=-1)
                            ax.contourf(X,Y,Z, [0., chi2[1]], colors=prior_cc, alpha=alphas[0],zorder=-1)
                        if isinstance(pp,priors.Uniform):
                            ax.axvspan(pp.lower[j],pp.upper[j], color=prior_cc, alpha=alphas[0],zorder=-1)
                            ax.axhspan(pp.lower[i],pp.upper[i], color=prior_cc, alpha=alphas[0],zorder=-1)
                      if plot_fid:
                        ax.scatter(val_fid[j],val_fid[i],color=fid_c,zorder=1,s=60)
                      ax.scatter(val_true[j],val_true[i],color=tru_c,zorder=1,s=25)
              else:
                  if plot_priors:
                    if isinstance(pp,priors.TruncatedGaussian):
                      ax.axvspan((pp.mean[i] - pp.C[i,i]), (pp.mean[i] + pp.C[i,i]), color=prior_cc, alpha=alphas[0],zorder=-1)
                      ax.axvspan((pp.mean[i] - 2*pp.C[i,i]), (pp.mean[i] + 2*pp.C[i,i]), color=prior_cc, alpha=alphas[1],zorder=-1)
                    if isinstance(pp,priors.Uniform):   
                      ax.axvspan(pp.lower[i],pp.upper[i], color=prior_cc, alpha=alphas[1],zorder=-1)
                  if plot_fid:
                    ax.axvline(val_fid[i],color=fid_c,lw=3)
                  ax.axvline(val_true[i],color=tru_c,lw=1)
              xtl = ax.get_xticklabels()
              ax.set_xticklabels(xtl, rotation=45)
          plt.tight_layout()
          plt.subplots_adjust(hspace=0, wspace=0)

          if savefig:
              plt.savefig(filename)
          if show_plot:
              plt.show()
          else:
              plt.close()
