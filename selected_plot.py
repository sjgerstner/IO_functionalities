import pickle
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = "serif"

with open('results/meta-llama/Llama-3.2-3B/data.pickle', 'rb') as f:
    data = pickle.load(f)

def wcos_plot(layer_list, arrangement, data):
    fig, axs = plt.subplots(arrangement[0], arrangement[1],
                        sharex=True,
                        sharey=True,
                        constrained_layout=True,
                        )
    fig.set_figwidth(4.5)
    fig.set_figheight(2)
    axs_list = axs.ravel().tolist()
    for i,layer in enumerate(layer_list):
        ax=axs_list[i]
        ax.set_xlim(-1.,1.)
        ax.set_ylim(-1.,1.)
        ax.set_aspect(1, share=True)
        scatter = ax.scatter(
            data["gateout"][layer], data["linout"][layer], c=data["gatelin"][layer],
            vmin=-1., vmax=1., linewidths=0, s=.25, rasterized=True
            )
        ax.set_title(f"Layer {layer}", fontsize=10)
    fig.supxlabel("$cos(w_{gate}, w_{out})$", fontsize=10)
    fig.supylabel("$cos(w_{in}, w_{out})$", fontsize=10)
    fig.colorbar(scatter, ax=axs_list, label="$cos(w_{gate}, w_{in})$")

    #fig.suptitle(model_name, fontsize=10)

    return fig, axs

layer_list = [0,14,27]
arrangement = (1,3)
MODEL_NAME= 'meta-llama/Llama-3.2-3B'
fig, axs = wcos_plot(layer_list, arrangement, data)
fig.savefig(f'results/{MODEL_NAME}/selected.pdf')
