# from .chaos_theory import *
# from .convergent_fusion import *
# from .divergent_expansion import *
# from .emergent_synthesis import *
# from .fractal_thinking import *
# from .infinite_mind import *
# from .metaphysical_illumination import *
# from .morphogenesis import *
# from .non_linear_navigation import *
# from .organic_synthesis import *
# from .power_of_perspective import *
# from .eureka_elicitation import *
# from .synergetic_fusion import *
# from .paradoxical_reflection import *
# from .quantum_entanglement import *
# from .radical_collaboration import *


# import random
# from typing import List


# class SynthesisTechniqueManager:
#     def __init__(self):
#         self.synthesis_techniques = [
#             TheInfiniteMind(),
#             ChaosTheory(),
#             EurekaElicitation(),
#             NonLinearNavigation(),
#             Morphogenesis(),
#             MetaphysicalIllumination(),
#             ConvergentFusion(),
#             DivergentExpansion(),
#             ParadoxicalReflection(),
#             QuantumEntanglement(),
#             RadicalCollaboration(),
#             SynergeticFusion(),
#             OrganicSynthesis(),
#             ThePowerOfPerspective(),
#             EmergentSynthesis(),
#         ]

#     def get_random_synthesis_technique_name(self) -> str:
#         return random.choice(self.get_synthesis_technique_names())

#     def get_synthesis_technique(self, name: str) -> SynthesisTechnique:
#         for synthesis_technique in self.synthesis_techniques:
#             if synthesis_technique.technique_name == name:
#                 return synthesis_technique
#         raise ValueError(f"Unknown synthesis technique {name}")

#     def get_synthesis_technique_names(self) -> list[str]:
#         return [
#             synthesis_technique.technique_name
#             for synthesis_technique in self.synthesis_techniques
#         ]

#     def get_synthesis_technique_epithets(self) -> List[str]:
#         return [
#             synthesis_technique.epithet
#             for synthesis_technique in self.synthesis_techniques
#         ]

#     def get_synthesis_technique_imperatives(self) -> List[str]:
#         return [
#             synthesis_technique.imperative
#             for synthesis_technique in self.synthesis_techniques
#         ]

#     def get_synthesis_technique_prompts(self) -> List[str]:
#         return [
#             synthesis_technique.prompts
#             for synthesis_technique in self.synthesis_techniques
#         ]

#     def create_synthesis_technique(self, name: str) -> SynthesisTechnique:
#         return self.get_synthesis_technique(name)


# #     def execute(self, chain_tree: IChainTree, node: Chain, **kwargs) -> None:
# #         try:
# #             max_branches = kwargs.get("max_branches", None)

# #             # Extract user-defined functions for scoring prompts, determining positions, choosing prompt type, dynamic branching, and user feedback.
# #             score_func = kwargs.get("score_func", None)
# #             position_func = kwargs.get("position_func", None)
# #             prompt_type_func = kwargs.get("prompt_type_func", None)
# #             branching_func = kwargs.get("branching_func", None)
# #             user_feedback_func = kwargs.get("user_feedback_func", None)

# #             # Set the default order of prompt types.
# #             prompt_type_order = ["branching_options", "dynamic_prompts"]

# #             # Calculate the middle of the coordinate space for the new chains.
# #             middle_coord = len(chain_tree.get_chains()) // 2

# #             print(
# #                 f"Executing synthesis technique {self.technique_name} with max_branches={max_branches}, score_func={score_func}, position_func={position_func}, prompt_type_func={prompt_type_func}, branching_func={branching_func}, user_feedback_func={user_feedback_func}"
# #             )
# #             for prompt_type in prompt_type_order:
# #                 # Get the options for the current prompt type.
# #                 options = self.get_synthesis_technique(prompt_type).get_options()

# #                 # If no valid options are found, log and return.
# #                 if not options:
# #                     log_handler(
# #                         f"No options found for prompt type {prompt_type} in {self.get_synthesis_technique_names()}"
# #                     )
# #                     return

# #                 # If a prompt type function is provided, use it to adaptively choose the prompt type.
# #                 if prompt_type_func is not None:
# #                     prompt_type = prompt_type_func(options, chain_tree, node)

# #                 # Determine the number of branches to be created.
# #                 num_branches = (
# #                     branching_func(node, chain_tree) if branching_func else len(options)
# #                 )

# #                 # Create a list of all prompts, along with their scores if a scoring function was provided.
# #                 all_prompts = [(prompt_type, option) for option in options]
# #                 if score_func is not None:
# #                     all_prompts = [
# #                         (score_func(prompt_type, option), prompt_type, option)
# #                         for prompt_type, option in all_prompts
# #                     ]
# #                     all_prompts.sort(
# #                         reverse=True
# #                     )  # Sort prompts in descending order of score.
# #                     all_prompts = all_prompts[
# #                         :num_branches
# #                     ]  # Only keep the top scoring prompts.

# #                 if not all_prompts:
# #                     log_handler("No valid prompts after scoring and sorting.")
# #                     return

# #                 for prompt in all_prompts:
# #                     if (
# #                         max_branches is not None
# #                         and len(chain_tree.get_chains()) >= max_branches
# #                     ):
# #                         log_handler("Maximum number of branches reached.")
# #                         return

# #                     if score_func is not None:
# #                         score, prompt_type, option = prompt
# #                     else:
# #                         prompt_type, option = prompt

# #                     # Compute the coordinates for the new chain.
# #                     if position_func is not None:
# #                         x_coord, y_coord, z_coord = position_func(
# #                             middle_coord,
# #                             node.coordinate.y,
# #                             prompt_type_order,
# #                             prompt_type,
# #                         )
# #                     else:
# #                         x_coord = middle_coord
# #                         y_coord = node.coordinate.y
# #                         z_coord = node.coordinate.z + prompt_type_order.index(
# #                             prompt_type
# #                         )

# #                     # Check if a chain with the same content already exists in the chain tree.
# #                     for chain in chain_tree.get_chains():
# #                         if chain.content == option:
# #                             log_handler("Skipping creation of duplicate chain.")
# #                             continue

# #                     # Add a new chain to the chain tree.

# #                     new_chain = chain_tree.add_chain(
# #                         chain_type="assistant",
# #                         id=f"{node.id}-{len(chain_tree.get_chains()) + 1}",
# #                         content=Content(option),
# #                         coordinate=Coordinate(x=x_coord, y=y_coord, z=z_coord),
# #                         metadata=None,
# #                     )

# #                     # Add a link between the new chain and the initial node.
# #                     chain_tree.add_link(
# #                         chain_a_id=node.id,
# #                         chain_b_id=new_chain.id,
# #                         metadata=None,
# #                     )

# #                     # If a user feedback function is provided, allow the user to review and approve the new chain.
# #                     if user_feedback_func is not None:
# #                         approved = user_feedback_func(new_chain, chain_tree)
# #                         if not approved:
# #                             log_handler(
# #                                 "User did not approve the new chain. Removing it from the chain tree."
# #                             )
# #                             chain_tree.remove_chain(new_chain.id)

# #         except Exception as e:
# #             log_handler(f"Error in synthesis technique {self.technique_name}: {e}")
# #             raise e

# #         return chain_tree


# if __name__ == "__main__":
#     manager = SynthesisTechniqueManager()

#     # get all synthesis techniques

#     print(manager.get_synthesis_technique_names())
