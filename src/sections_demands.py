import re
import pandas as pd
import logging
from typing import Optional, Tuple

from siancedb.models import (
    Session,
    SessionWrapper,
    SiancedbDemand,
    SiancedbLetter,
    SiancedbSection,
)
from siancedb.pandas_writer import chunker
from siancedb.sql_logger import logger_step



def normalize_text(text):
    """
    Replace most of special characters in a text, and turn it to lowercase

    Args:
        text (str): a text that may includes accents, diacritics or common (not all) french special characters

    Returns:
        str: a lowercase text where most of common french special characters have been replaced
    """
    text_lower = (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("’", "'")
        .replace("é", "e")
        .replace("É", "e")
        .replace("Ê", "e")
        .replace("È", "e")
        .replace("Ç", "c")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("ù", "u")
        .replace("ï", "i")
        .replace("ç", "c")
        .replace("ô", "o")
        .lower()
    )
    return text_lower

from datetime import datetime
logger = logger_step(step='BUILD DEMANDS')


def find_positions_blocks_containing_demands(
    positions_blocks, positions_demands, text, n_previous_blocks=0
):
    """
    For every demand start, find the start of the corresponding block (a block can be a paragraph, a sentence,
     or any other division).
     If there is several demands in the same block, don't repeat the block in the output (so the two demands in the
     same block are considered to be a single one)

     If parameter `n_previous_blocks` is not 0, every block containing a demand will be merged with (at most) the
     n  "not-special" blocks (=blocks that does not contain demand themselves) that precedes the block with demand,
     in order to bring more context information.

     Three conditions must be checked in order to merge a block containing a demand with some not-special blocks:
        (1) a block with a demand cannot be merged with not-special blocks before the very first block
        (2) a block with a demand cannot be merged with a block containing another demand
        (3) a block with a demand cannot be merged with a not special block already merged with a block with demand
     In particular, if there is only m (m < n) not-special blocks between a demand 1 and a demand 2, only these m
     blocks will be added in the context of demand 2.


    For the example
    > position_blocks = [1, 6, 14, 68, 90, 115, 136]
    > position_demands = [3, 5, 13, 67, 120]
    > n_previous_blocks = 0
    the output is
    > [(1, 6), (6, 14), (14, 68), (115,136)]
    reminder: the demands indicated at the positions 3 and 5 are in the same block, so they are considered to be merely
    one demand

    For the example (quite tricky)
    > position_blocks = [1, 6, 14, 68, 90, 115, 136]
    > position_demands = [3, 5, 13, 67, 120]
    > n_previous_blocks = 1
    the output is
    > [(1, 6), (6, 14), (14, 68), (90,136)]

    """
    assert (
        isinstance(n_previous_blocks, int) and n_previous_blocks >= 0
    ), "n_previous_block must be a non-negative integer"
    positions_blocks.sort(key=lambda x: x)
    positions_demands.sort(key=lambda x: x)

    iter_blocks = 0
    iter_demands = 0
    blocks_containing_demands = []
    merged_blocks = []

    # Remark: normally position_blocks[-1] should be the length of the text
    while iter_blocks < len(positions_blocks) - 1:
        if iter_demands == len(positions_demands):
            break

        block_beginning = positions_blocks[iter_blocks]

        # condition 1: a block can not be merged with blocks before the first block (!)
        shift_iter = max(0, iter_blocks - n_previous_blocks)
        merged_block_beginning = positions_blocks[shift_iter]

        block_ending = positions_blocks[iter_blocks + 1]
        # condition 2: if we have already met another block with demand, we can not merge with him
        # if we have met such a block, the length of `blocks_containing_demands` is > 0
        if len(blocks_containing_demands) > 0:
            # as the lists of demands were sorted, the previous block with demand if necessarily the last one added
            # in `blocks_containing_demands`
            (
                previous_block_with_demand_beginning,
                previous_block_with_demand_ending,
            ) = blocks_containing_demands[-1]
        else:
            # if we have never met a block with demand before, we can artificially set this variable to 0
            previous_block_with_demand_ending = 0

        # the position of the demand (assuming a demand is not initially covering several blocks at the same time)
        demand_beginning = positions_demands[iter_demands]

        # if there is a demand in the block, store the bounds of the block in the `blocks_containing_demands` list
        if block_beginning <= demand_beginning < block_ending:
            # check there is not already a demand in the same block. if this the case, the 2 demands must be "merged"
            # to "merge" them, it is necessary and sufficient just to store only one time the block containing them
            if (block_beginning, block_ending) not in blocks_containing_demands:
                blocks_containing_demands.append((block_beginning, block_ending))
                # condition (3) two "merged blocks" are not allowed to intersect
                # if we had never met a block with demand, previous_block_with_demand_ending=0, so the max() works
                merged_block_beginning = max(
                    merged_block_beginning, previous_block_with_demand_ending
                )
                merged_block_beginning = max(merged_block_beginning, demand_beginning)

                #Remove traling jump lines
                num_newlines = len(text[merged_block_beginning: block_ending]) - len(text[merged_block_beginning: block_ending].rstrip('\n'))
                merged_blocks.append((merged_block_beginning, block_ending - num_newlines))



            iter_demands += 1
        else:
            iter_blocks += 1

    return merged_blocks


def divide_one_letter(text: str):
    """
    Given one letter, cut it into sections `Synthesis` (synthese d'une inspection),
    `demands` (demandes d'actions correctives), `information` (demandes de compléments),
    `observations` (observations de l'inspecteur). The parts `introduction` (sentence before synthesis)
    and `conclusion` (sentences after the closing formula) are cut away and not returned by the function

    Warning for developers: in this function, separators should not contain accents
    or unusal punctuations, as the implemented matching ignores them

    Args:
        text (str): the body of the letter (lettre de suite)

    Raises:
        Exception: "The letter is not following any known format".
        This exception is deprecated in this code version

    Returns:
        str, str, str, str, int (or None), int (or None), int (or None), int (or None), int (or None), int (or None),
         int (or None), int (or None): the subtexts corresponding to `synthesis`, `demands`, `information`,
         `observations` sections, and the starting and ending characters of each section (equals to None when section
         is missing)
    """
    # try:
    text_lower = normalize_text(text)
    sep_synthesis = [
        r"[\r\n|\n|\\n]+[(i\. )|(1\. )|(a\. )|(1 \- )]*s[a-z]{1,2}these de (l'insp[a-z]{3}ion|la visite)[\.| |:]*",
        r"1[\-|\.| ]*synthese de l'inspection[\.| |:]*",
        r"[\r\n|\n|\\n]+synthese des inspections[\.| |:]*",
        r"[\r\n|\n|\\n]+synthese du contrôle[\.| |:]*",
        r"[\r\n|\n|\\n]+[ ]*synthese[\.| |:]*[\r\n|\n|\\n]+",
        r"[\r\n|\n|\\n]+i. appreciation globale",
    ]
    sep_synthesis = "|".join(sep_synthesis)

    sep_demands = [
        r"[1|2|a|b|ii][ |\.|\-|\|\/)]+demande[s]* d'action[s]* corrective[s]*[\.| |:]*",
        r"[1|2|a|b|ii][ |\.|\-|\|\/)]+demande[s]* d'a[a-z]{1,2}ion[s]* co[a-z]{5,6}ve[s]*[\.| |:]*",
        r"[\r\n|\n|\\n]+demande[s]* d'action[s]* corrective[s]*[\.| |:]*",
        r"demande[s]* d'action[s]* corrective[s]*[\.| |:]*[\r\n|\n|\\n]+",
        r"[\r\n|\n|\\n]+a[\.| \-]{0,1} demandes[ :]{0,1}[\r\n|\n|\\n]+",
        r"[\r\n|\n|\\n]+[1|2|a|b|ii][ |\.|\-|\|\/)]+demandes d'action[s]*[\r\n|\n|\\n]+",
        r"[\r\n|\n|\\n]+[a\. ]*description des ecarts[\r\n|\n|\\n]+",
        r"[\r\n|\n|\\n]+a. actions correctives[\r\n|\n|\\n]+",
        r"[\r\n|\n|\\n]+a[0-9]{0,1}. actions correctives[\r\n|\n|\\n]+",
        r"[\r\n|\n|\\n]+a. (demande[s]* de )*mise[s]* en conformite a la reglementation",
        r"ii[ |\.|\-|\|\/)]+demande[s]* portant sur des ecarts[\.| |:]*",
        r"ii[ |\.|\-|\|\/)]+demande[s]* d'engagements[\.| |:]*",
        r"[1|2|a|b|ii][ |\.|\-|\|\/)]+principales constatations et demandes",
    ]
    sep_demands = "|".join(sep_demands)

    sep_demandes_prioritaires = [
        r"(1|i)\. demandes (a|à) traiter prioritairement",
    ]
    sep_demandes_prioritaires = "|".join(sep_demandes_prioritaires)      

    sep_information = [
        r"[a|b|c|ii|iii|2|3][ |\.|\-|\|\/)]+demande[s]*( d'information[s]*)* complementaire[s]*[\.| |:]*",
        r"[a|b|c|ii|iii|2|3][ |\.|\-|\|\/)]+demande[s]* d'information[s]*[\.| |:]*",
        r"[a|b|c|ii|iii|2|3][ |\.|\-|\|\/)]+demande[s]* de complement[s]*( d'information[s]*){0,1}[\.| |:]*",
        r"[\r\n|\n|\\n]+(demande[s]* de ){0,1}complement[s]* d'information[s]*[\.| |:]*",
        r"[\r\n|\n|\\n]+b[\.] d'informations complementaires[\.| |:]*",
        r"[a|b|c|ii|iii|2|3][ |\.|\-|\|\/)]+complement[s]* d'information[s]*[\.| |:]*",
        r"[ a-z]*complement[s]* d'informations[\.| |:]*[\r\n|\n|\\n]+",
        r"[ a-z]*complement[s]* d'information[s]*[\.| |:]*[\r\n|\n|\\n]+",
        r"[a|b|c|ii|iii|2|3][ |\.|\-|\|\/)]+demande[s]* de justification et de positionnement[\.| |:]*",
    ]
    sep_information = "|".join(sep_information)

    sep_autres_demandes = [
        r"(2|ii)\. autres demandes"
    ]
    sep_autres_demandes = "|".join(sep_autres_demandes)      


    sep_observations = [
        r"[\r\n|\n|\\n]+[b|c|2|iv][ |\.|\-|\|\/)]+observation[s]*[\.| |:]*",
        r"[2|iv][ |\.|\-|\|\/)]+observation[s]*[\.| |:]*",
        r"[\r\n|\n|\\n]+[ ]*observation[s]*[\.| |:]*[\r\n|\n|\\n]+",
        r"iii\. constats ou observations[ n('|’)appelant pas de r(e|é)ponse]*[ (a|à) l(’|')asn]*",
    ]
    sep_observations = "|".join(sep_observations)

    sep_conclusion = [
        r"je vous prie de trouver, ci-joint, les axes d'amelioration identifies au cours de l'inspection",
        r"(vous voudrez bien|je vous saurai gre de bien vouloir) me f[a-z]{2}re part",
    ]
    sep_conclusion = "|".join(sep_conclusion)

    """
    The letter is parsed from the end to the beginning, because a section goes from the title
    of a section to the beginning of the next present section
    """

    # `start_next_section` is a tmp variable, giving the starting point of the next present section
    start_next_section = len(text)

    search_conclusion = re.search(sep_conclusion, text_lower)
    if search_conclusion is not None:
        # store where the closing formula starts
        start_title_conclusion = search_conclusion.start()
        # the conclusion part comes after (the beginning of) the closing formula
        start_conclusion = start_title_conclusion
        text_conclusion = text[start_conclusion:]
        # stores in `start_next_section` the beginning of conclusion part
        start_next_section = start_title_conclusion
    else:
        text_conclusion = ""
        start_conclusion = None

    search_observations = re.search(sep_observations, text_lower)
    if search_observations is not None:
        # store where the SECTION TITLE of observations starts and ends
        start_title_observations = search_observations.start()
        end_title_observations = search_observations.end()
        start_observations = (
            end_title_observations  # skip the title of "Observations" section
        )
        end_observations = start_next_section
        text_observations = text[start_observations:end_observations]
        # stores in `start_next_section` the beginning of observations part
        start_next_section = start_title_observations
    else:
        text_observations = ""
        start_observations, end_observations = None, None

    search_autres_demandes = re.search(sep_autres_demandes, text_lower)
    if search_autres_demandes is not None:
        # store where the SECTION TITLE of demands of informations starts and ends
        start_title_autres_demandes = search_autres_demandes.start()
        end_title_autres_demandes = search_autres_demandes.end()
        start_autres_demandes = end_title_autres_demandes  # skip the title of "Demandes d'Informations" section
        end_autres_demandes = start_next_section
        text_autres_demandes = text[start_autres_demandes:end_autres_demandes]
        # stores in `start_next_section` the beginning of demands of information part
        start_next_section = start_title_autres_demandes
    else:
        text_autres_demandes = ""
        start_autres_demandes, end_autres_demandes = None, None

    search_information = re.search(sep_information, text_lower)
    if search_information is not None:
        # store where the SECTION TITLE of demands of informations starts and ends
        start_title_information = search_information.start()
        end_title_information = search_information.end()
        start_information = end_title_information  # skip the title of "Demandes d'Informations" section
        end_information = start_next_section
        text_information = text[start_information:end_information]
        # stores in `start_next_section` the beginning of demands of information part
        start_next_section = start_title_information
    else:
        text_information = ""
        start_information, end_information = None, None


    search_demands = re.search(sep_demands, text_lower)
    if search_demands is not None:
        # store where the SECTION TITLE of demands of actions starts and ends
        start_title_demands = search_demands.start()
        end_title_demands = search_demands.end()
        start_demands = (
            end_title_demands  # skip the title of "Demandes d'Actions" section
        )
        end_demands = start_next_section
        text_demands = text[start_demands:end_demands]
        # stores in `start_next_section` the beginning of demands of information part
        start_next_section = start_title_demands
    else:
        text_demands = ""
        start_demands, end_demands = None, None

    search_demandes_prioritaires = re.search(sep_demandes_prioritaires, text_lower)
    if search_demandes_prioritaires is not None:
        # store where the SECTION TITLE of demands of actions starts and ends
        start_title_demands = search_demandes_prioritaires.start()
        end_title_demands = search_demandes_prioritaires.end()
        start_demandes_prioriaires = (
            end_title_demands  # skip the title of "Demandes d'Actions" section
        )
        end_demandes_prioriaires = start_next_section
        text_demandes_prioriaires = text[start_demandes_prioriaires:end_demandes_prioriaires]
        # stores in `start_next_section` the beginning of demands of information part
        start_next_section = start_title_demands
    else:
        text_demandes_prioriaires = ""
        start_demandes_prioriaires, end_demandes_prioriaires = None, None

    search_synthesis = re.search(sep_synthesis, text_lower)
    if search_synthesis is not None:
        # store where the SECTION TITLE of synthesis starts and ends
        start_title_synthesis = search_synthesis.start()
        end_title_synthesis = search_synthesis.end()
        start_synthesis = (
            end_title_synthesis  # skip the title of "Demandes d'Actions" section
        )
        end_synthesis = start_next_section
        text_synthesis = text[start_synthesis:end_synthesis]
        # stores in `start_next_section` the beginning of demands of information part
        start_next_section = start_title_synthesis
    else:
        text_synthesis = ""
        start_synthesis, end_synthesis = None, None

    text_introduction = text[:start_next_section]

    return (
        text_synthesis,
        text_demands,
        text_demandes_prioriaires,
        text_autres_demandes,
        text_information,
        text_observations,
        start_synthesis,
        end_synthesis,
        start_demands,
        end_demands,
        start_demandes_prioriaires,
        end_demandes_prioriaires,
        start_autres_demandes,
        end_autres_demandes,
        start_information,
        end_information,
        start_observations,
        end_observations,
    )

    # except Exception as e:
    #     print(text)
    #     print(e)
    #     raise Exception("The letter is not following any known format")


def divide_letters(letters_df: pd.DataFrame, strip_texts=True):
    """
    Given a dataframe of letters, divide them in sections `synthesis`, `demands`, `information`,
    `observations`, and return a dataframe with these (four) columns and with a (fifth) column `id_letter`

    Args:
        letters_df (pandas.DataFrame): tables of letters, whose bodies are in the column `text`
            and with an index `id_letter`
        strip_texts (bool): if True (default), the output texts are stripped (starting and ending blanks are cut)
            Nevertheless, if these blanks are removed, the returned starting characters may be wrong
            TODO: must be changed, in order to remove this boolean and fix the starting characters

    Returns:
        pandas.DataFrame: dataframe with columns `id_letter`, `synthesis`, `demands`, `information`,
        `observations`
    """
    data = []
    for id_letter, row in letters_df.set_index("id_letter").iterrows():
        text = str(row["text"])
        (
            text_synthesis,
            text_demands,
            text_information,
            text_observations,
            start_synthesis,
            end_synthesis,
            start_demands,
            end_demands,
            start_information,
            end_information,
            start_observations,
            end_observations,
        ) = divide_one_letter(text)
        data.append(
            [
                id_letter,
                text_synthesis,
                text_demands,
                text_information,
                text_observations,
                start_synthesis,
                end_synthesis,
                start_demands,
                end_demands,
                start_information,
                end_information,
                start_observations,
                end_observations,
            ]
        )
    return pd.DataFrame(
        data=data,
        columns=[
            "synthesis",
            "demands",
            "information",
            "observations",
            "conclusion",
            "start_synthesis",
            "end_synthesis",
            "start_demands",
            "end_demands",
            "start_information",
            "end_information",
            "start_observations",
            "end_observations",
        ],
    )


def extract_demands_one_letter(text: str, n_previous_blocks=1):
    """
    Slow function for tests only. In production the useful function is `build_sections_demands_one_letter`

    Given a dataframe of letters, extract the demands from the section `demands`(demandes d'actions correctives) and
    from the section `information` (demandes d'information complémentaire)
    `observations`, and return a dataframe with the columns `start`, `end`, `sentence`, `priority`(1 for demandes
    d'actions correctives, and 2 for demandes d'information complémentaire) with an `id_letter`

    Args:
        text (str): the content of the letter read from letters table (it comes after the cleaning)
        n_previous_blocks (int):
    Returns:
        pandas.DataFrame: dataframe with at least columns `start`, `end`, `priority`
    """

    (
        text_synthesis,
        text_demands,
        text_information,
        text_observations,
        start_synthesis,
        end_synthesis,
        start_demands,
        end_demands,
        start_information,
        end_information,
        start_observations,
        end_observations,
    ) = divide_one_letter(text)

    (
        absolute_positions_demands_a,
        absolute_positions_information_b,
    ) = get_positions_demands_a_b(
        start_demands, start_information, text_demands, text_information
    )

    """
    sentencizer = prepare_sentencizer()
    (
        absolute_positions_sentences_a,
        absolute_positions_sentences_b,
    ) = get_positions_sentences_a_b(
        sentencizer, start_demands, start_information, text_demands, text_information
    )
    bounds_blocks_demands_a = find_positions_blocks_containing_demands(
        absolute_positions_sentences_a, absolute_positions_demands_a
    )

    bounds_blocks_demands_b = find_positions_blocks_containing_demands(
        absolute_positions_sentences_b, absolute_positions_information_b
    )

    """

    (
        absolute_positions_paragraphs_a,
        absolute_positions_paragraphs_b,
    ) = get_positions_paragraphs_a_b(
        start_demands, start_information, text_demands, text_information
    )

    bounds_blocks_demands_a = find_positions_blocks_containing_demands(
        absolute_positions_paragraphs_a,
        absolute_positions_demands_a,
        n_previous_blocks=n_previous_blocks,
    )

    bounds_blocks_demands_b = find_positions_blocks_containing_demands(
        absolute_positions_paragraphs_b,
        absolute_positions_information_b,
        n_previous_blocks=n_previous_blocks,
    )

    demands_table = []
    # demands_table is a list of list [sentence demand start, sentence demand end, type(can be one or 2), text]
    for bounds in bounds_blocks_demands_a:
        demands_table.append(
            [
                bounds[0],
                bounds[1],
                1,
                text_demands[bounds[0] - start_demands : bounds[1] - start_demands],
            ]
        )
    for bounds in bounds_blocks_demands_b:
        demands_table.append(
            [
                bounds[0],
                bounds[1],
                2,
                text_information[
                    bounds[0] - start_information : bounds[1] - start_information
                ],
            ]
        )
    return pd.DataFrame(
        data=demands_table, columns=["start", "end", "priority", "sentence"]
    )


def get_positions_sentences_a_b(
    sentencizer, start_demands, start_information, text_demands, text_information
):
    """
    Given the text of sections A and B, and their starting characters, compute the positions
    of every sentences in these sections (these sentences may be actual demands or not)
    The returned positions are calculated from the beginning of the letter

    Args:
        sentencizer ([type]): a NLP object that cut text in sentences
        start_demands (int | None): the position of the start of the section A (None if the section doesn't exist)
        start_information (int | None): the position of the start of the section B (None if the section doesn't exist)
        text_demands (str): the text of the section A of the letter
        text_information (str): the text of the section B of the letter

    Returns:
        list[int], list[int]:
            the lists of positions of the starting character of sentences in sections A and B
    """
    absolute_positions_sentences_a = [
        start_demands + sent.start_char for sent in sentencizer(text_demands).sents
    ]  # possibly empty if the section is empty
    # add the length of the section, to have the bounds of every sentence including the last one
    absolute_positions_sentences_a.append(start_demands + len(text_demands))
    absolute_positions_sentences_b = [
        start_information + sent.start_char
        for sent in sentencizer(text_information).sents
    ]
    # add the length of the section, to have the bounds of every sentence including the last one
    absolute_positions_sentences_b.append(start_information + len(text_information))
    return absolute_positions_sentences_a, absolute_positions_sentences_b


def get_positions_paragraphs_a_b(
    start_demands, start_information, text_demands, text_information
):
    """
    Given the text of sections A and B, and their starting characters, compute the positions
    of every paragraphs in these sections (the function cuts too short paragraphs)
    The returned positions are calculated from the beginning of the letter

    Args:
        start_demands (int | None): the position of the start of the section A (None if the section doesn't exist)
        start_information (int | None): the position of the start of the section B (None if the section doesn't exist)
        text_demands (str): the text of the section A of the letter
        text_information (str): the text of the section B of the letter

    Returns:
        list[int], list[int]:
            the lists of positions of the starting character of paragraphs (excluding very short paragraphs)
    """
    # if there are too few characters between two "\n", it means some paragraphs are nonsense. Ignore them
    paragraph_min_length = 8
    # p = re.compile(r"\n\n|\\n\\n|\n|\\n")
    p = re.compile(r"\n\n|\\n\\n")

    # the first paragraph begins with the beginning of the section (if the section is not empty)
    if start_demands is not None:
        absolute_positions_paragraphs_a = [start_demands]
        for newline_sign in p.finditer(text_demands):
            new_paragraph = start_demands + newline_sign.end()
            absolute_positions_paragraphs_a.append(new_paragraph)

        # add the position of the start of the last paragraph (potentially empty) of the section
        absolute_positions_paragraphs_a.append(start_demands + len(text_demands))

        # check the length of the paragraph to filter out very short paragraphs
        # if there are too few characters between two "\n", it means some paragraphs are nonsense. Ignore them
        # example: if min_length = 5 and if the positions are [0, 7, 9, 20], it becomes [0, 7, 20]
        absolute_positions_paragraphs_a = [
            absolute_positions_paragraphs_a[k]
            for k in range(len(absolute_positions_paragraphs_a) - 1)
            if (
                absolute_positions_paragraphs_a[k + 1]
                - absolute_positions_paragraphs_a[k]
            )
            >= paragraph_min_length
        ]
        absolute_positions_paragraphs_a.append(start_demands + len(text_demands))
    else:
        absolute_positions_paragraphs_a = []

    if start_information is not None:
        absolute_positions_paragraphs_b = [start_information]
        for newline_sign in p.finditer(text_information):
            new_paragraph = start_information + newline_sign.end()
            absolute_positions_paragraphs_b.append(new_paragraph)
        # add the position of the start of the last paragraph (potentially empty) of the section
        absolute_positions_paragraphs_b.append(
            start_information + len(text_information)
        )

        # check the length of the paragraph to filter out very short paragraphs
        # if there are too few characters between two "\n", it means some paragraphs are nonsense. Ignore them
        absolute_positions_paragraphs_b = [
            absolute_positions_paragraphs_b[k]
            for k in range(0, len(absolute_positions_paragraphs_b) - 1)
            if (
                absolute_positions_paragraphs_b[k + 1]
                - absolute_positions_paragraphs_b[k]
            )
            >= paragraph_min_length
        ]
        absolute_positions_paragraphs_b.append(
            start_information + len(text_information)
        )
    else:
        absolute_positions_paragraphs_b = []

    return absolute_positions_paragraphs_a, absolute_positions_paragraphs_b


def get_positions_demands_a_b(
    start_demands, start_information, text_demands, text_information
):
    """
    Given the text of sections A and B, and their starting characters, compute the starting positions
    of every demand pattern (see regex) in these sections.
    The returned positions are calculated from the beginning of the letter

    Args:
        start_demands (int | None): the position of the start of the section A (None if the section doesn't exist)
        start_information (int | None): the position of the start of the section B (None if the section doesn't exist)
        text_demands (str): the text of the section A of the letter
        text_information (str): the text of the section B of the letter

    Returns:
        list[int], list[int]:
            the lists of positions of the starting character of demand patterns in sections A and B
    """
    pattern_demand = [
        r"Demande (A|B)\d\. *(:)?je vous (demande|invite)",
        r"Demande (A|B)\d\. *(:)?asn vous (demande|invite)",
        r"Demande (A|B)\d\. *(:)?l('|’)asn vous (demande|invite)",
        r"(A|B)\d\. *(:)?je vous demande",
        r"(A|B)\d\. *(:)?asn vous demande",
        r"(A|B)\d\. *(:)?l'asn vous demande",
        r"je vous demande",
        r"l'asn vous demande",
        r"asn vous demande",
        r"asn vous invite",
        r"je vous invite",
        r"demande i\.",
        r"demande ii.",
        r"demande n°i\.",
        r"demande n°ii\.",
        r"demande ii"
    ]
    pattern_demand = re.compile("|".join(pattern_demand), re.IGNORECASE)
    absolute_positions_demands_a = [
        m.start() + start_demands for m in pattern_demand.finditer(text_demands)
    ]
    # absolute_positions_demands_a = [
    #     m.start() for m in pattern_demand.finditer(text_demands)
    # ]
    absolute_positions_information_b = [
        m.start() + start_information for m in pattern_demand.finditer(text_information.lower())
    ]
    return absolute_positions_demands_a, absolute_positions_information_b


def build_sections_demands_one_letter(db: Session,
    letter: SiancedbLetter,
):
    """
    Given a letter, extract all its sections and demands, and return list of SiancedbSection and SiancedbDemand objects

    Args:
        letter (SiancedbLetter): an instance of letter model
        n_previous_blocks (int):
        sentencizer ([type]): a NLP object that cut text in sentences

    Returns:
        list[SiancedbSection], list[SiancedbDemand]:the list of sections and demands mentioned in the letter content
    """
    # First, delete existing sections and demands for this letter
    existing_sections = db.query(SiancedbSection).filter(SiancedbSection.id_letter == letter.id_letter).all()
    for section in existing_sections:
        db.delete(section)

    existing_demands = db.query(SiancedbDemand).filter(SiancedbDemand.id_letter == letter.id_letter).all()
    for demand in existing_demands:
        db.delete(demand)    
    db.commit()


    (
            text_synthesis,
            text_demands,
            text_demandes_prioriaires,
            text_autres_demandes,
            text_information,
            text_observations,
            start_synthesis,
            end_synthesis,
            start_demands,
            end_demands,
            start_demandes_prioriaires,
            end_demandes_prioriaires,
            start_autres_demandes,
            end_autres_demandes,
            start_information,
            end_information,
            start_observations,
            end_observations,
    ) = divide_one_letter(str(letter.text))

    db_sections = []
    if len(text_synthesis) > 0:
        db_sections.append(
            SiancedbSection(
                id_letter=letter.id_letter,
                priority=0,
                start=start_synthesis,
                end=end_synthesis,
            )
        )

    # OLD LETTERS (< mai 2022)
    if len(text_demands) > 0:
        db_sections.append(
            SiancedbSection(
                id_letter=letter.id_letter,
                priority=1,
                start=start_demands,
                end=end_demands,
            )
        )
    if len(text_information) > 0:
        db_sections.append(
            SiancedbSection(
                id_letter=letter.id_letter,
                priority=2,
                start=start_information,
                end=end_information,
            )
        )

    #NEW LETTERS (>mai 2022)
    if len(text_demandes_prioriaires) > 0:
        db_sections.append(
            SiancedbSection(
                id_letter=letter.id_letter,
                priority=4,
                start=start_demandes_prioriaires,
                end=end_demandes_prioriaires,
            )
        )
    if len(text_autres_demandes) > 0:
        db_sections.append(
            SiancedbSection(
                id_letter=letter.id_letter,
                priority=5,
                start=start_autres_demandes,
                end=end_autres_demandes,
            )
        )

    
    ### COMMON TO OLD AND NEW
    if len(text_observations) > 0:
        db_sections.append(
            SiancedbSection(
                id_letter=letter.id_letter,
                priority=3,
                start=start_observations,
                end=end_observations,
            )
        )

    absolute_positions_paragraphs_a = []
    absolute_positions_paragraphs_b = []
    absolute_positions_demands_a = []
    absolute_positions_information_b = []
    if len(text_demandes_prioriaires) > 0 or len(text_autres_demandes) > 0:
        (
            absolute_positions_demands_a,
            absolute_positions_information_b,
        ) = get_positions_demands_a_b(
            start_demandes_prioriaires, start_autres_demandes, text_demandes_prioriaires, text_autres_demandes
        )
    elif len(text_demands) > 0 or len(text_information) > 0:
        (
            absolute_positions_demands_a,
            absolute_positions_information_b,
        ) = get_positions_demands_a_b(
            start_demands, start_information, text_demands, text_information
        )

    if len(text_demandes_prioriaires) > 0 or len(text_autres_demandes) > 0:
        (
            absolute_positions_paragraphs_a,
            absolute_positions_paragraphs_b,
        ) = get_positions_paragraphs_a_b(
            start_demandes_prioriaires, start_autres_demandes, text_demandes_prioriaires, text_autres_demandes
        )
    elif len(text_demands) > 0 or len(text_information) > 0:
        (
        absolute_positions_paragraphs_a,
        absolute_positions_paragraphs_b,
        ) = get_positions_paragraphs_a_b(
            start_demands, start_information, text_demands, text_information
        )

    bounds_blocks_demands_a = find_positions_blocks_containing_demands(
        absolute_positions_paragraphs_a,
        absolute_positions_demands_a,
        letter.text,
        n_previous_blocks=0,
    )

    bounds_blocks_demands_b = find_positions_blocks_containing_demands(
        absolute_positions_paragraphs_b,
        absolute_positions_information_b,
        letter.text,
        n_previous_blocks=0,
    )

    db_demands = []
    for bounds_a in bounds_blocks_demands_a:
        db_demands.append(
            SiancedbDemand(
                start=bounds_a[0],
                end=bounds_a[1],
                priority=1,
                id_letter=letter.id_letter,
            )
        )
    for bounds_b in bounds_blocks_demands_b:
        db_demands.append(
            SiancedbDemand(
                start=bounds_b[0],
                end=bounds_b[1],
                priority=2,
                id_letter=letter.id_letter,
            )
        )

    return db_sections, db_demands


def build_sections_demands(db: Session, redo_all=False, redo_some=None):
    """
    For all letters for which no sections and demands have been extracted, extract them and save them in the database
    Nota Bene: A letter necessarily contains a section or a demand

    Args:
        db (Session): a Session to connect to the database
        n_previous_blocks (int):
    """
    dquery_any = (
        db.query(SiancedbLetter)
        .filter(~SiancedbLetter.sections.any())
        .filter(~SiancedbLetter.demands.any())
    )
    if redo_all is False and redo_some is None:
        dquery = (
            db.query(SiancedbLetter)
            .filter(SiancedbLetter.date_demands_update.is_(None))
        )
    elif redo_all is False and redo_some is not None:
        print("Will redo only letters with names in", redo_some)
        dquery = (
            db.query(SiancedbLetter)
            .filter(SiancedbLetter.name.in_(redo_some))
        )    
    elif redo_all is True:
        dquery = db.query(SiancedbLetter)

    letters = dquery.all()
    n_documents_any = dquery_any.count()
    n_documents = dquery.count()


    # just check this len() function does not empty letter pseudo-generator
    letters_count = 0

    logger.info(f" ** {n_documents_any} letters don't have demands or sections. {n_documents} letters have never been checked for demands or sections **")
    chunk_size = 100
    n_demands, n_sections = 0, 0
    for i_chunk, chunk_letters in enumerate(chunker(100, letters)):
        logger.info(f"Starting chunk {i_chunk + 1} / {n_documents // chunk_size + 1}")
        chunk_sections, chunk_demands = [], []
        for letter in chunk_letters:
            (
                one_letter_db_sections,
                one_letter_db_demands,
            ) = build_sections_demands_one_letter(db, 
                letter,
            )
            chunk_sections.extend(one_letter_db_sections)
            chunk_demands.extend(one_letter_db_demands)
            letter.date_demands_update = datetime.now()
            db.add(letter)
        n_demands += len(chunk_demands)
        n_sections += len(chunk_sections)
        db.add_all(chunk_sections)
        db.add_all(chunk_demands)

        letters_count += 100
        db.commit()
        logger.info(f"Chunk {i_chunk + 1} / {n_documents // chunk_size + 1} commited.")
    logger.info(f"** {n_demands} demands added and {n_sections} sections added **")    


if __name__ == "__main__":
    # list_1 = [1, 6, 14, 68, 136]
    # list_2 = [9, 13, 67, 136, 137]
    # print(find_positions_blocks_containing_demands(list_1, list_2))
    with SessionWrapper() as db:
        # build_sections_demands(db)
        build_sections_demands(db, redo_some=["INSSN-LYO-2021-0469"])
