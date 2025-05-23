# %% [markdown]
# # Definition plot

# %% [markdown]
# You may need to run the rcParams cell and the plotting cell twice each to get the right format.

# %% [markdown]
# ### Setup

#%%
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# %%
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = "serif"

# %%
mydict = {
    "enrichment": ([-.9,.9],[.9,.9],[-1,1], "P"),
    "conditional enrichment": (0,.9,0, "X"),
    "depletion": ([-.9,.9],[-.9,-.9],[1,-1], "s"),
    "conditional depletion": (0,-.9,0, "D"),
    "proportional change": ([-.9,.9],[0,0],[0,0],"*"),
    "orthogonal output": (0,0,0, "o"),
}

# %%
marker_dict = {}
for key, value in mydict.items():
    marker_dict[key] = mlines.Line2D([], [], color='black', ls='', label=key, marker=value[-1])

# %%
plt.rcParams['figure.figsize'] = [2,1.75]

# %%
fig, ax = plt.subplots()
ax.set_xlim(-1.,1.)
ax.set_ylim(-1.,1.)

scatter = {}
for key, val in mydict.items():
    scatter[key] = ax.scatter(
        val[0], val[1], c=val[2], marker=val[3], label=key,
        vmin=-1., vmax=1., linewidths=0, s=100
    )
ax.set_xlabel("$cos(w_{gate}, w_{out})$")
ax.set_ylabel("$cos(w_{in}, w_{out})$")
fig.colorbar(scatter["orthogonal output"], ax=ax, label="$cos(w_{gate}, w_{in})$")

legend = plt.legend(handles=[value for _key,value in marker_dict.items()],
                    bbox_to_anchor=(2,1),
                    markerscale = 1.5,
                    )
fig.savefig("plots/prototypes_var.pdf", bbox_inches='tight')
fig.show()
