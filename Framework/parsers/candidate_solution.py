import random
from parsers import grammar_parser

def legal_productions(gram, root, productions, method='random', depth_limit=100):
    """
    Returns the available production choices for a node given a specific
    depth limit. We recommend that you do not change the values of the method
    and depth_limit parameters.

    :param method: A string specifying the desired tree derivation method.
    Current methods are "random" or "full". There are some methods that use
    the option "full", but we are always going to use the option "random".
    :param depth_limit: The overall depth limit of the desired tree from the
    current node.
    :param root: The root of the current node.
    :param productions: The full list of production choices from the current
    root node.
    :return: The list of available production choices based on the specified
    derivation method.
    """

    # Get all information about root node
    root_info = gram.non_terminals[root]

    if method == "random":
        # Randomly build a tree.

        if not depth_limit:
            # There is no depth limit, any production choice can be used.
            available = productions

        elif depth_limit > gram.max_arity + 1:
            # If the depth limit is greater than the maximum arity of the
            # grammar, then any production choice can be used.
            available = productions

        elif depth_limit < 0:
            # If we have already surpassed the depth limit, then list the
            # choices with the shortest terminating path.
            available = root_info['min_path']

        else:
            # The depth limit is less than or equal to the maximum arity of
            # the grammar + 1. We have to be careful in selecting available
            # production choices lest we generate a tree which violates the
            # depth limit.
            available = [prod for prod in productions if prod['max_path'] <=
                         depth_limit - 1]

            if not available:
                # There are no available choices which do not violate the depth
                # limit. List the choices with the shortest terminating path.
                available = root_info['min_path']

    elif method == "full":
        # Build a "full" tree where every branch extends to the depth limit.

        if not depth_limit:
            # There is no depth limit specified for building a Full tree.
            # Raise an error as a depth limit HAS to be specified here.
            s = "representation.derivation.legal_productions\n" \
                "Error: Depth limit not specified for `Full` tree derivation."
            raise Exception(s)

        elif depth_limit > gram.max_arity + 1:
            # If the depth limit is greater than the maximum arity of the
            # grammar, then only recursive production choices can be used.
            available = root_info['recursive']

            if not available:
                # There are no recursive production choices for the current
                # rule. Pick any production choices.
                available = productions

        else:
            # The depth limit is less than or equal to the maximum arity of
            # the grammar + 1. We have to be careful in selecting available
            # production choices lest we generate a tree which violates the
            # depth limit.
            available = [prod for prod in productions if prod['max_path'] ==
                         depth_limit - 1]

            if not available:
                # There are no available choices which extend exactly to the
                # depth limit. List the NT choices with the longest terminating
                # paths that don't violate the limit.
                available = [prod for prod in productions if prod['max_path']
                             < depth_limit - 1]

    return available


def get_random_cand(grammar, maxdepth, old_genome=None):
    """Generate a random candidate (genome and string) from a grammar
    """
    father = None
    # we use this inner helper function to do the work.
    # it "accumulates" the genome as it runs.

    def _random_ind(gram, genome, tree_genome, father, depth, s=None, name=None):
        """Recursively create a genome. gram is a grammar, genome a dict
        (initially empty), tree_genome is a list presenting the expansion
        of the grammar productions depth an integer giving maximum depth. 
        s is the current symbol (None tells us to use the start symbol.) 
        name is the name-in-progress."""
        if s is None:
            s = gram.start_rule["symbol"]
            name = tuple()
        elif s in gram.terminals:
            return s

        rule = gram.rules[s]

        if old_genome and name in old_genome:

            # A valid entry was found in old_genome. Apply mod rule as
            # part of repair: it will have no effect if the value is
            # already in the right range.
            gi = old_genome[name] % len(rule['choices'])
            prod = rule['choices'][gi]

        else:

            # No valid entry was found, so choose a production from
            # among those which are legal (because their min depth to
            # finish recursion is less than or equal to max depth
            # minus our current depth).
            productions = gram.rules[s]
            available = legal_productions(gram, s, productions['choices'],
                                          method="random", depth_limit=depth)
            prod = random.choice(available)  # choose production
            gi = productions['choices'].index(prod)  # find its index
	

        genome[name] = gi
        candidate_tree = [(maxdepth - depth), s,
                          prod['choice'][0]['symbol'], father]
        father = s
        tree_genome.append(candidate_tree)

        # Join together all the strings for this production. For
        # terminals just use the string itself. For non-terminals,
        # recurse: decrease depth, pass in the symbol, and append to
        # the naming scheme according to the choice we made in the
        # current call. Recall that each symbol s is a dict:
        # s["symbol"] is the symbol itself, s["type"] is 'T' or 'NT'.
        return "".join((s["symbol"] if s["type"] == "T"
                        else
                        _random_ind(gram,
                                    genome,
                                    tree_genome,
                                    father,
                                    depth - 1,
                                    s["symbol"],
                                    name + ((gi, i),)))
                       for i, s in enumerate(prod['choice']))

    genome = {}
    tree_genome = []
    s = _random_ind(grammar, genome, tree_genome, father, maxdepth, None, None)

    # return genome, s, tree_genome
    return tree_genome, s




def get_random_candidates(grammar_file, number_of_candidates, seed, att):
    """
    It gets a random set of candidates from the grammar.
    """
    candidates = list()
    gr = grammar_parser.Grammar(grammar_file)
    random.seed(a=seed, version=2)

    for i in range(number_of_candidates):        
        cand = get_random_cand(gr, 100, None)
        updated_cand = convert_candidate(cand, att)
        candidates.append(updated_cand)

    return candidates



def convert_candidate(cand, att):
    """
    It converts the random ranges
    into the string and the tree 
    to actual values.
    """
    float_str = 'RANDFLOAT'
    int_str = 'RANDINT'
    att_str = 'RANDATT'

    tree_rep = cand[0]
    str_rep = cand[1]
    str_rep = str_rep.replace(', ', ',')
    str_rep = str_rep.replace(' ,', ',')
    str_rep = str_rep.replace(' , ', ',')

    str_rep_array = str_rep.split(' ')
    str_rep = ''
    split_str_len = len(str_rep_array)

    pos=0

    for i in range(0, (split_str_len)):
        if(str_rep_array[i].startswith(float_str)):
            str_rep_array[i] = get_float_into_range(str_rep_array[i])
            conv_output = deal_with_tree(tree_rep, str_rep_array[i], pos, float_str)
            tree_rep = conv_output[0]
            pos = conv_output[1]
        elif(str_rep_array[i].startswith(int_str)):
            str_rep_array[i] = get_int_into_range(str_rep_array[i], att)
            conv_output = deal_with_tree(tree_rep, str_rep_array[i], pos, int_str)        
            tree_rep = conv_output[0]
            pos = conv_output[1]             
        elif(str_rep_array[i].startswith(att_str)):
            str_rep_array[i] = get_int_into_range(str_rep_array[i], att)
            conv_output = deal_with_tree(tree_rep, str_rep_array[i], pos, att_str)        
            tree_rep = conv_output[0]
            pos = conv_output[1]   

       
        str_rep += str_rep_array[i] + ' '

    return tree_rep, str_rep



def deal_with_tree(tree, value, pos, start_str):
    """
    It deals with the produced tree, translating a 
    range to a specific random value.
    """
    tree_len = len(tree)
    current_pos = pos

    for i in range(pos, tree_len):
        if(tree[i][2].startswith(start_str)):
            tree[i][2] = value
            current_pos = i
            break    

    return tree, current_pos


def get_float_into_range(prodct):
    """
    It gets a float value into a range.
    """
    range_len = len(prodct)
    interval = prodct[10:(range_len-1)]
    split_rg = interval.split(',')
   
    float_value = random.uniform(float(split_rg[0]), float(split_rg[1]))

    return str(float_value)


def get_int_into_range(prodct, att):
    """
    It gets a integer value into a range.
    It can also get a integer value into a range
    that depends on the number of attributes of 
    the input dataset.
    """
    range_len = len(prodct)
    interval = prodct[8:(range_len-1)]
    interval = interval.strip()
    split_rg = interval.split(',')
    int_value = -1

    if(prodct[0:7]=='RANDINT'):
        int_value = random.randint(int(split_rg[0]), int(split_rg[1]))
    else:
        int_value = random.randint(int(split_rg[0]), (att-1))

    return str(int_value)
