
import copy

from .tucker2 import Tucker2DecomposedLayer
from .cp3 import CP3DecomposedLayer
import numpy as np


def get_compressed_model(MODEL_NAME,model, ranks=[], layer_names=[], layer_names_bn=[], decompositions=[],
                         pretrained=None,
                         vbmf_weaken_factor=None,
                         return_ranks=False):
    compressed_model = copy.deepcopy(model)

    new_ranks = copy.deepcopy(np.array(ranks, dtype=object))


    for i, (rank, layer_name, layer_name_bn, decomposition) in enumerate(zip(ranks, layer_names, layer_names_bn,decompositions)):



        if rank is not None :
            print('Decompose layer', layer_name)
            subm_names = layer_name.strip().split('.')
            subm_names_bn = layer_name_bn.strip().split('.')



            layer = compressed_model.__getattr__(subm_names[0])
            layer_bn = compressed_model.__getattr__(subm_names_bn[0])

            for s in subm_names[1:]:
                layer = layer.__getattr__(s)


            for s in subm_names_bn[1:]:
                layer_bn = layer_bn.__getattr__(s)


            if decomposition == 'tucker2':

                decomposed_layer = Tucker2DecomposedLayer(MODEL_NAME, layer, subm_names[-1],
                                                               rank,
                                                              pretrained=pretrained)





            elif decomposition == 'cp3':
                decomposed_layer = CP3DecomposedLayer(layer, subm_names[-1], rank,
                                                      pretrained=pretrained)












            try:
                new_ranks[i] = decomposed_layer.ranks
            except:
                new_ranks[i] = decomposed_layer.rank


            print('\t new rank: ', new_ranks[i])




            if len(subm_names) > 1:
                m = compressed_model.__getattr__(subm_names[0])

                for s in subm_names[1:-1]:
                    m = m.__getattr__(s)
                m.__setattr__(subm_names[-1], decomposed_layer.new_layers)
            else:
                compressed_model.__setattr__(subm_names[-1], decomposed_layer.new_layers)





        else:
            print('Skip layer', layer_name)

    if return_ranks:
        return compressed_model
    else:
        return compressed_model
