class Recombination:
  @staticmethod
  def run(offs, weights, mu_rank_elite_size):
    return offs.dot(weights[: mu_rank_elite_size])
