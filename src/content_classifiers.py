import re
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ContentType(Enum):
    SECTION_HEADER = "section_header"
    DEMAND = "demand"
    REGULAR = "regular"


class SectionType(Enum):
    SYNTHESIS = "synthesis"
    DEMANDS = "demands"
    DEMANDES_PRIORITAIRES = "demandes_prioritaires"
    AUTRES_DEMANDES = "autres_demandes"
    INFORMATION = "information"
    OBSERVATIONS = "observations"
    INTRODUCTION = "introduction"
    CONCLUSION = "conclusion"


def normalize_text(text: str) -> str:
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


class ContentClassifier:
    """Classifier for document blocks to identify sections and demands"""
    
    def __init__(self):
        self.section_patterns = {
            SectionType.SYNTHESIS: [
                r"(?:^|\n+)[(i\. )|(1\. )|(a\. )|(1 \- )]*s[a-z]{1,2}these de (l'insp[a-z]{3}ion|la visite)[\.| |:]*",
                r"1[\-|\.| ]*synthese de l'inspection[\.| |:]*",
                r"(?:^|\n+)synthese des inspections[\.| |:]*",
                r"(?:^|\n+)synthese du contrôle[\.| |:]*",
                r"(?:^|\n+)[ ]*synthese[\.| |:]*(?:\n+|$)",
                r"(?:^|\n+)i. appreciation globale",
            ],
            SectionType.DEMANDS: [
                r"[1|2|a|b|ii][ |\.|\-|\|\/)]+demande[s]* d'action[s]* corrective[s]*[\.| |:]*",
                r"[1|2|a|b|ii][ |\.|\-|\|\/)]+demande[s]* d'a[a-z]{1,2}ion[s]* co[a-z]{5,6}ve[s]*[\.| |:]*",
                r"(?:^|\n+)demande[s]* d'action[s]* corrective[s]*[\.| |:]*",
                r"demande[s]* d'action[s]* corrective[s]*[\.| |:]*(?:\n+|$)",
                r"(?:^|\n+)a[\.| \-]{0,1} demandes[ :]{0,1}(?:\n+|$)",
                r"(?:^|\n+)[1|2|a|b|ii][ |\.|\-|\|\/)]+demandes d'action[s]*(?:\n+|$)",
                r"(?:^|\n+)[a\. ]*description des ecarts(?:\n+|$)",
                r"(?:^|\n+)a. actions correctives(?:\n+|$)",
                r"(?:^|\n+)a[0-9]{0,1}. actions correctives(?:\n+|$)",
                r"(?:^|\n+)a. (demande[s]* de )*mise[s]* en conformite a la reglementation",
                r"ii[ |\.|\-|\|\/)]+demande[s]* portant sur des ecarts[\.| |:]*",
                r"ii[ |\.|\-|\|\/)]+demande[s]* d'engagements[\.| |:]*",
                r"[1|2|a|b|ii][ |\.|\-|\|\/)]+principales constatations et demandes",
            ],
            SectionType.DEMANDES_PRIORITAIRES: [
                r"(1|i)\. demandes (a|à) traiter prioritairement",
            ],
            SectionType.INFORMATION: [
                r"[a|b|c|ii|iii|2|3][ |\.|\-|\|\/)]+demande[s]*( d'information[s]*)* complementaire[s]*[\.| |:]*",
                r"[a|b|c|ii|iii|2|3][ |\.|\-|\|\/)]+demande[s]* d'information[s]*[\.| |:]*",
                r"[a|b|c|ii|iii|2|3][ |\.|\-|\|\/)]+demande[s]* de complement[s]*( d'information[s]*){0,1}[\.| |:]*",
                r"(?:^|\n+)(demande[s]* de ){0,1}complement[s]* d'information[s]*[\.| |:]*",
                r"(?:^|\n+)b[\.] d'informations complementaires[\.| |:]*",
                r"[a|b|c|ii|iii|2|3][ |\.|\-|\|\/)]+complement[s]* d'information[s]*[\.| |:]*",
                r"[ a-z]*complement[s]* d'informations[\.| |:]*(?:\n+|$)",
                r"[ a-z]*complement[s]* d'information[s]*[\.| |:]*(?:\n+|$)",
                r"[a|b|c|ii|iii|2|3][ |\.|\-|\|\/)]+demande[s]* de justification et de positionnement[\.| |:]*",
            ],
            SectionType.AUTRES_DEMANDES: [
                r"(2|ii)\. autres demandes"
            ],
            SectionType.OBSERVATIONS: [
                r"(?:^|\n+)[b|c|2|iv][ |\.|\-|\|\/)]+observation[s]*[\.| |:]*",
                r"[2|iv][ |\.|\-|\|\/)]+observation[s]*[\.| |:]*",
                r"(?:^|\n+)[ ]*observation[s]*[\.| |:]*(?:\n+|$)",
                r"iii\. constats ou observations[ n('|')appelant pas de r(e|é)ponse]*[ (a|à) l('|')asn]*",
            ],
            SectionType.CONCLUSION: [
                r"je vous prie de trouver, ci-joint, les axes d'amelioration identifies au cours de l'inspection",
                r"(vous voudrez bien|je vous saurai gre de bien vouloir) me f[a-z]{2}re part",
            ]
        }
        
        self.demand_patterns = [
            r"Demande (A|B)\d\. *(:)?je vous (demande|invite)",
            r"Demande (A|B)\d\. *(:)?asn vous (demande|invite)",
            r"Demande (A|B)\d\. *(:)?l('|')asn vous (demande|invite)",
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
        
        # Compile patterns for efficiency
        self.compiled_section_patterns = {}
        for section_type, patterns in self.section_patterns.items():
            self.compiled_section_patterns[section_type] = re.compile(
                "|".join(patterns), re.IGNORECASE
            )
        
        self.compiled_demand_pattern = re.compile(
            "|".join(self.demand_patterns), re.IGNORECASE
        )
    
    def classify_block(
        self, 
        content: str, 
        current_section: Optional[SectionType] = None
    ) -> Tuple[ContentType, Optional[SectionType], Optional[int]]:
        """
        Classify a block's content.
        
        Args:
            content: The text content of the block
            current_section: The current section context (if known)
            
        Returns:
            Tuple of (content_type, section_type, demand_priority)
        """
        normalized_content = normalize_text(content)
        
        # Check if it's a section header
        for section_type, pattern in self.compiled_section_patterns.items():
            if pattern.search(normalized_content):
                return (ContentType.SECTION_HEADER, section_type, None)
        
        # Check if it's a demand
        if self.compiled_demand_pattern.search(normalized_content):
            # Determine priority based on current section
            priority = None
            if current_section in [SectionType.DEMANDS, SectionType.DEMANDES_PRIORITAIRES]:
                priority = 1
            elif current_section in [SectionType.INFORMATION, SectionType.AUTRES_DEMANDES]:
                priority = 2
            return (ContentType.DEMAND, current_section, priority)
        
        # Otherwise it's regular content
        return (ContentType.REGULAR, current_section, None)
    
    def is_letter_de_suite(self, blocks_content: List[str]) -> bool:
        """
        Check if the document appears to be a 'lettre de suite' based on section patterns.
        
        Args:
            blocks_content: List of block contents to analyze
            
        Returns:
            True if document contains typical letter sections
        """
        full_text = " ".join(blocks_content).lower()
        normalized_full = normalize_text(full_text)
        
        # Check for at least 2 different section types
        found_sections = set()
        for section_type, pattern in self.compiled_section_patterns.items():
            if section_type != SectionType.INTRODUCTION and pattern.search(normalized_full):
                found_sections.add(section_type)
        
        return len(found_sections) >= 2