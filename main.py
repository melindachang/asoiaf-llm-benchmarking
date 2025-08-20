#!/usr/bin/env python3
"""
Multi-Model Danish Legend Analyzer
Sends same stories to Claude, GPT, and Gemini for comparison
"""

import json
import os
import time
import logging
import argparse
import webbrowser
import chardet
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading

# API clients
import anthropic
import openai
import google.generativeai as genai
import requests  # For Together AI API

# Schema validation
try:
    from jsonschema import validate, ValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    logging.warning("jsonschema not available - install with: pip install jsonschema")

# Optional dependencies for enhanced features
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.info("Visualization libraries not available - install with: pip install matplotlib seaborn pandas")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load API keys from local file:
try:
    with open('api_keys.json', 'r') as f:
        api_keys = json.load(f)
        for key, value in api_keys.items():
            os.environ[key] = value
    logger.info("API keys loaded from api_keys.json")
except FileNotFoundError:
    logger.warning("api_keys.json not found, using environment variables")

def check_api_keys():
    """Check that required API keys are available"""
    required_keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]
    optional_keys = ["TOGETHER_API_KEY"]
    missing_keys = []
    
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        logger.error(f"Missing API keys: {missing_keys}")
        raise ValueError(f"Missing required API keys: {missing_keys}")
    else:
        logger.info("All required API keys found")
    
    for key in optional_keys:
        if os.getenv(key):
            logger.info(f"Optional API key found: {key}")

@dataclass
class ModelConfig:
    """Configuration for each AI model"""
    provider: str
    model_name: str
    api_key_env: str
    max_tokens: int = 4000
    temperature: float = 0.1
    top_p: Optional[float] = None
    delay_between_calls: float = 1.0
    max_retries: int = 3

@dataclass
class AnalysisConfig:
    """Main configuration for the analysis"""
    models: List[ModelConfig]
    output_dir: str = "./multi_model_results"
    concurrent_models: bool = True
    save_individual_responses: bool = True
    create_comparison: bool = True
    create_concatenated_output: bool = True
    calculate_agreement_scores: bool = True
    create_html_summary: bool = False
    open_after_run: bool = False

class PromptManager:
    """Manages the standardized prompts"""
    
    @staticmethod
    def get_refined_prompt() -> str:
        return """You are an expert folklorist analyzing Danish 19th-century legends. Provide both ontological tagging and structural analysis using the refined guidelines below.

ONTOLOGICAL CLASSIFICATION SYSTEM:
**People Classes:** Aristocrat/Royal, Bailiff, Beggar/Poor person, Boy, Children, Cunning Folk, Farmer, Farmwife, Farmhand/Shepherd, Foreigner, Girl, Man, Midwife, Minister, Parish Clerk, Robber/Thief, Sailor/Fisherman, Soldier, Strong Man/Hero, Wanderer, Witch, Woman, Unspecified

**Place Classes:** Barn, Bridge, Cemetery, Church, Farm, Field, Forest/Woods, Heath, Inn, Manor/Castle, Market, Meadow, Mill, Mound, Ocean/Sea, Pond/Lake, Road, Stream/River, Swamp, Town/Village

**Tools, Items and Conveyances:** Bell, Boat, Books, Food and Drink, Metal, Plow, Tools, Treasure/Money, Wagon, Unspecified

**Supernatural Beings:** Changeling, Devil, Elf, Ghost/Revenant, Giant, Mare, Merfolk, Mound folk, Nightraven, Nisse, Troll, Werewolf, Unspecified, Lindorm/Basilisk, Dragon

**Animals:** Cat, Cow, Dog, Farm animal, Hare, Horse, Ox, Serpent/Snake, Sheep, Swine, None/Unspecified, Insects, Wild, Lindorm/Basilisk

**Actions or Events:** Church ceremony, Conjuring, Dancing, Death, Disease, Drowning, Feast/Celebration, Gambling, Haunting, Injury, Kidnapping, Midwifery, Murder, Music, Oath, Punishment, Suicide, Theft, Transaction, Witchcraft, Work, Pregnancy and birth, Seduction

**Time, Season, Weather:** Sunday, Monday, Thursday, Friday, Saturday, Morning, Noon, Evening, Night, Midnight, Spring, Summer, Winter, Fall, Holiday, Skt. Hans, Christmas, New Years, Easter, Snow, Rain, Moon

**REFINED RESOLUTION CATEGORIES:**
- **Positive:** Threat resolved favorably, goals achieved, beneficial outcome
- **Negative:** Threat realized, harm occurs, goals thwarted  
- **Neutral:** No harm but no benefit, status quo maintained
- **Unresolved:** Story ends without addressing the threat/disruption
- **Transformative:** Situation fundamentally changed (neither clearly positive nor negative)

**REFINED THREAT CATEGORIES:**
- **Physical Safety:** Bodily harm, death, injury
- **Economic Security:** Loss of property, livelihood, resources
- **Social Standing:** Reputation, community position, relationships  
- **Spiritual/Supernatural:** Soul, salvation, supernatural retribution
- **Psychological:** Mental state, sanity, peace of mind
- **Cultural/Normative:** Violation of social rules, taboos

STRUCTURAL ANALYSIS (Labov-Waletzky modified by Tangherlini):
1. **Abstract:** Brief summary or introductory remark (null if not present)
2. **Orientation:** Who, what, where, when
3. **Complicating Action: Disruption:** Specific threat/disruption and threatening agent
4. **Complicating Action: Strategy:** What insiders decide as strategy against threat
5. **Evaluation:** Commentary within narrative (not analyst interpretation)
6. **Resolution:** Outcome using refined categories above
7. **Aftermath:** Long-term consequences mentioned in story

ANNOTATION RULES:
- Every ontological tag must have clear textual evidence
- No speculative or implied tags
- People Classes = only humans; Supernatural Beings = non-human sentient entities
- Resolution must match actual story outcome, not implications
- Use specific threat categories, not vague ones

Return ONLY a valid JSON object with this exact format:

{
  "story_text": "[original story]",
  "ontological_tags": {
    "people_classes": [],
    "place_classes": [],
    "tools_items_conveyances": [],
    "supernatural_beings": [],
    "animals": [],
    "actions_events": [],
    "time_season_weather": [],
    "resolution": [],
    "stylistics": []
  },
  "structural_analysis": {
    "abstract": "[text or null]",
    "orientation": {
      "who": [],
      "what": "[description]", 
      "where": "[location]",
      "when": "[time]"
    },
    "complicating_action_disruption": [],
    "threat_categories": [],
    "threatening_agent": "[description]",
    "complicating_action_strategy": [],
    "evaluation": "[text or null]",
    "resolution": "[using refined categories]",
    "aftermath": "[description or null]"
  },
  "notes": {
    "evidence_for_tags": "[brief justification for key tags]",
    "resolution_rationale": "[why this resolution category]"
  }
}

Story to analyze:
"""

class AIProvider:
    """Base class for AI providers"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.api_key = os.getenv(config.api_key_env)
        if not self.api_key:
            raise ValueError(f"Environment variable {config.api_key_env} not set")
    
    def analyze(self, prompt: str, story: str) -> Dict[str, Any]:
        raise NotImplementedError
    
    def extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from various response formats"""
        response_text = response_text.strip()
        
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        elif '```' in response_text and response_text.count('```') >= 2:
            response_text = response_text.split('```')[1]
        
        start_idx = response_text.find('{')
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(response_text[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            response_text = response_text[start_idx:end_idx]
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Response was: {response_text[:500]}...")
            raise

class ClaudeProvider(AIProvider):
    """Anthropic Claude provider"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def analyze(self, prompt: str, story: str) -> Dict[str, Any]:
        full_prompt = prompt + story
        
        params = {
            "model": self.config.model_name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": full_prompt}]
        }
        
        if self.config.top_p is not None:
            params["top_p"] = self.config.top_p
        
        response = self.client.messages.create(**params)
        response_text = response.content[0].text
        return self.extract_json_from_response(response_text)

class OpenAIProvider(AIProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def analyze(self, prompt: str, story: str) -> Dict[str, Any]:
        full_prompt = prompt + story
        
        params = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": "You are a folklorist analyzing Danish legends. Return only valid JSON."},
                {"role": "user", "content": full_prompt}
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        if self.config.top_p is not None:
            params["top_p"] = self.config.top_p
        
        response = self.client.chat.completions.create(**params)
        response_text = response.choices[0].message.content
        return self.extract_json_from_response(response_text)

class GeminiProvider(AIProvider):
    """Google Gemini provider"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(config.model_name)
    
    def analyze(self, prompt: str, story: str) -> Dict[str, Any]:
        full_prompt = prompt + story
        
        config_params = {
            "max_output_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        
        if self.config.top_p is not None:
            config_params["top_p"] = self.config.top_p
        
        generation_config = genai.types.GenerationConfig(**config_params)
        
        response = self.model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        response_text = response.text
        return self.extract_json_from_response(response_text)

class TogetherProvider(AIProvider):
    """Together AI provider for open-source models like Llama"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_url = "https://api.together.xyz/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def analyze(self, prompt: str, story: str) -> Dict[str, Any]:
        full_prompt = prompt + story
        
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": "You are a folklorist analyzing Danish legends. Return only valid JSON."},
                {"role": "user", "content": full_prompt}
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p
        
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=payload,
            timeout=120
        )
        
        if response.status_code != 200:
            raise Exception(f"Together API error: {response.status_code} - {response.text}")
        
        response_data = response.json()
        response_text = response_data["choices"][0]["message"]["content"]
        return self.extract_json_from_response(response_text)

class MultiModelAnalyzer:
    """Main analyzer that coordinates multiple AI models"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.prompt_manager = PromptManager()
        
        # Set up output directory structure
        Path(config.output_dir).mkdir(exist_ok=True)
        
        # Create subdirectories for each model
        for model_config in config.models:
            model_dir = Path(config.output_dir) / f"{model_config.provider}_output"
            model_dir.mkdir(exist_ok=True)
        
        # Create directories for comparisons and concatenated results
        Path(config.output_dir, "comparisons").mkdir(exist_ok=True)
        Path(config.output_dir, "concatenated").mkdir(exist_ok=True)
        Path(config.output_dir, "agreement_scores").mkdir(exist_ok=True)
        
        # Initialize providers
        self.providers = {}
        for model_config in config.models:
            if model_config.provider == "claude":
                self.providers[model_config.provider] = ClaudeProvider(model_config)
            elif model_config.provider == "openai":
                self.providers[model_config.provider] = OpenAIProvider(model_config)
            elif model_config.provider == "gemini":
                self.providers[model_config.provider] = GeminiProvider(model_config)
            elif model_config.provider == "together":
                self.providers[model_config.provider] = TogetherProvider(model_config)
            else:
                raise ValueError(f"Unknown provider: {model_config.provider}")
    
    def analyze_story_single_model(self, story: str, story_id: str, provider_name: str) -> Dict[str, Any]:
        """Analyze a story with a single model"""
        provider = self.providers[provider_name]
        model_config = next(c for c in self.config.models if c.provider == provider_name)
        
        prompt = self.prompt_manager.get_refined_prompt()
        
        for attempt in range(model_config.max_retries):
            try:
                result = provider.analyze(prompt, story)
                result["story_id"] = story_id
                result["provider"] = provider_name
                result["model_name"] = model_config.model_name
                result["analysis_timestamp"] = time.time()
                return result
            
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {provider_name} on story {story_id}: {e}")
                if attempt == model_config.max_retries - 1:
                    logger.error(f"All attempts failed for {provider_name} on story {story_id}")
                    return {
                        "story_id": story_id,
                        "provider": provider_name,
                        "error": str(e),
                        "analysis_timestamp": time.time()
                    }
                time.sleep(model_config.delay_between_calls * 2)
    
    def analyze_story_all_models(self, story: str, story_id: str) -> Dict[str, Dict[str, Any]]:
        """Analyze a story with all configured models"""
        results = {}
        
        if self.config.concurrent_models:
            with ThreadPoolExecutor(max_workers=len(self.providers)) as executor:
                future_to_provider = {
                    executor.submit(self.analyze_story_single_model, story, story_id, provider_name): provider_name
                    for provider_name in self.providers.keys()
                }
                
                for future in future_to_provider:
                    provider_name = future_to_provider[future]
                    try:
                        result = future.result(timeout=120)
                        results[provider_name] = result
                    except Exception as e:
                        logger.error(f"Error with {provider_name}: {e}")
                        results[provider_name] = {
                            "story_id": story_id,
                            "provider": provider_name,
                            "error": str(e)
                        }
        else:
            for provider_name in self.providers.keys():
                logger.info(f"Analyzing story {story_id} with {provider_name}")
                results[provider_name] = self.analyze_story_single_model(story, story_id, provider_name)
                
                model_config = next(c for c in self.config.models if c.provider == provider_name)
                time.sleep(model_config.delay_between_calls)
        
        return results
    
    def analyze_corpus(self, stories: List[Dict[str, str]]) -> None:
        """Analyze a corpus of stories with all models"""
        logger.info(f"Starting multi-model analysis of {len(stories)} stories")
        logger.info(f"Using models: {[f'{config.provider} (temp={config.temperature})' for config in self.config.models]}")
        
        all_results = {}
        
        for i, story_data in enumerate(stories, 1):
            story_id = story_data.get("id", f"story_{i:03d}")
            story_text = story_data.get("text", story_data.get("story", ""))
            
            if not story_text:
                logger.warning(f"No text found for story {story_id}")
                continue
            
            logger.info(f"Analyzing story {i}/{len(stories)}: {story_id}")
            
            results = self.analyze_story_all_models(story_text, story_id)
            all_results[story_id] = results
            
            if self.config.save_individual_responses:
                for provider_name, result in results.items():
                    model_dir = Path(self.config.output_dir) / f"{provider_name}_output"
                    output_file = model_dir / f"{story_id}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Completed analysis for story {story_id}")
        
        logger.info("Multi-model corpus analysis complete")

def load_stories_from_file(filepath: str) -> List[Dict[str, str]]:
    """Load stories from various file formats"""
    filepath = Path(filepath)
    
    if filepath.suffix.lower() == '.csv':
        import csv
        stories = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            sample = f.read(1024)
            f.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            
            reader = csv.DictReader(f, delimiter=delimiter)

            for i, row in enumerate(reader):
                # def safe_strip(value):
                #     if value is None:
                #         return ''
                #     elif isinstance(value, str):
                #         return value.strip()
                #     else:
                #         return str(value).strip()

                # clean_row = {safe_strip(k): safe_strip(v) for k, v in row.items()}                               
                # # clean_row = {k.strip(): v for k, v in row.items()}
                # Debug: Print raw values before safe_strip
                if i < 3:
                    print(f"Row {i} RAW data:")
                    for k, v in row.items():
                        print(f"  '{k}': type={type(v)}, value='{str(v)[:100]}...'")
                    print()

                # Your existing safe_strip and clean_row code
                def safe_strip(value):
                    if value is None:
                        return ''
                    elif isinstance(value, str):
                        return value.strip()
                    else:
                        return str(value).strip()

                clean_row = {safe_strip(k): safe_strip(v) for k, v in row.items()}

                # Debug: Print cleaned data
                if i < 3:
                    print(f"Row {i} CLEANED data:")
                    for k, v in clean_row.items():
                        print(f"  '{k}': type={type(v)}, length={len(v)}, value='{str(v)[:100]}...'")
                    print()

                story_text = None
                story_id = None
                
                # Look for text columns
                text_columns = ['text', 'story', 'content', 'narrative', 'legend', 'tale', 'translated_text']
                for col in text_columns:
                    if col in clean_row and clean_row[col].strip():
                        story_text = clean_row[col].strip()
                        break
                
                # Look for ID columns
                id_columns = ['id', 'story_id', 'legend_id', 'number', 'no', 'index']
                for col in id_columns:
                    if col in clean_row and clean_row[col]:
                        story_id = str(clean_row[col]).strip()
                        break
                
                if not story_id:
                    story_id = f"story_{i+1:03d}"
                
                if story_text:
                    story_dict = {"id": story_id, "text": story_text}
                    
                    # Add metadata
                    metadata_columns = ['source', 'location', 'date', 'narrator', 'collector', 'region', 'type', 'pub_info', 'danish_publication', 'corrected_text']
                    for col in metadata_columns:
                        if col in clean_row and clean_row[col]:
                            story_dict[col] = clean_row[col]
                    
                    stories.append(story_dict)
        
        logger.info(f"Loaded {len(stories)} stories from CSV file")
        return stories
    
    elif filepath.suffix.lower() == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
    
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

def preview_csv_structure(filepath: str, num_rows: int = 3) -> None:
    """Preview the structure of a CSV file"""
    import csv
    
    filepath = Path(filepath)
    if not filepath.suffix.lower() == '.csv':
        print(f"File {filepath} is not a CSV file")
        return
    
    print(f"PREVIEW OF CSV FILE: {filepath}")
    print("=" * 50)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        sample = f.read(1024)
        f.seek(0)
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample).delimiter
        print(f"Detected delimiter: '{delimiter}'")
        print()
        
        reader = csv.DictReader(f, delimiter=delimiter)
        
        columns = reader.fieldnames
        print("COLUMNS FOUND:")
        for i, col in enumerate(columns):
            print(f"  {i+1:2d}. '{col.strip()}'")
        print()
        
        print(f"SAMPLE ROWS (first {num_rows}):")
        print("-" * 30)
        
        for i, row in enumerate(reader):
            if i >= num_rows:
                break
            
            print(f"Row {i+1}:")
            for col, value in row.items():
                value_preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                print(f"  {col.strip():15}: {value_preview}")
            print()

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="Multi-Model Danish Legend Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input_file", help="Input CSV file")
    parser.add_argument("-o", "--output-dir", default="./multi_model_results", help="Output directory")
    parser.add_argument("-m", "--models", nargs="+", choices=["claude", "openai", "gemini", "together"], 
                       default=["claude", "openai", "gemini"], help="AI models to use")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature")
    parser.add_argument("--top-p", type=float, help="Top-p parameter")
    parser.add_argument("--max-tokens", type=int, default=4000, help="Max tokens")
    parser.add_argument("--concurrent", action="store_true", help="Run concurrently")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between calls")
    parser.add_argument("--story-limit", type=int, help="Limit number of stories")
    parser.add_argument("--text-column", help="Text column name")
    parser.add_argument("--id-column", help="ID column name")
    parser.add_argument("--preview-only", action="store_true", help="Preview CSV only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    
    return parser

def create_model_configs_from_args(args: argparse.Namespace) -> List[ModelConfig]:
    """Create model configurations from arguments"""
    configs = []
    
    model_mapping = {
        "claude": {"provider": "claude", "model_name": "claude-3-5-sonnet-20241022", "api_key_env": "ANTHROPIC_API_KEY"},
        "openai": {"provider": "openai", "model_name": "gpt-4o", "api_key_env": "OPENAI_API_KEY"},
        "gemini": {"provider": "gemini", "model_name": "gemini-1.5-pro", "api_key_env": "GEMINI_API_KEY"},
        "together": {"provider": "together", "model_name": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "api_key_env": "TOGETHER_API_KEY"}
    }
    
    for model_name in args.models:
        if model_name in model_mapping:
            model_info = model_mapping[model_name]
            config = ModelConfig(
                provider=model_info["provider"],
                model_name=model_info["model_name"],
                api_key_env=model_info["api_key_env"],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                delay_between_calls=args.delay
            )
            configs.append(config)
    
    return configs

def main():
    """Main function"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.preview_only:
        if Path(args.input_file).suffix.lower() == '.csv':
            preview_csv_structure(args.input_file)
        return
    
    check_api_keys()
    
    model_configs = create_model_configs_from_args(args)
    if not model_configs:
        logger.error("No valid models specified")
        return
    
    config = AnalysisConfig(
        models=model_configs,
        output_dir=args.output_dir,
        concurrent_models=args.concurrent
    )
    
    if args.dry_run:
        logger.info("DRY RUN MODE - Configuration:")
        logger.info(f"Input file: {args.input_file}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Models: {[f'{cfg.provider}({cfg.model_name})' for cfg in model_configs]}")
        logger.info(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
        logger.info(f"Concurrent: {args.concurrent}, Delay: {args.delay}s")
        if args.text_column or args.id_column:
            logger.info(f"CSV columns - Text: {args.text_column}, ID: {args.id_column}")
        return
    
    try:
        logger.info(f"Loading stories from {args.input_file}")
        stories = load_stories_from_file(args.input_file)
        
        if args.story_limit:
            stories = stories[:args.story_limit]
            logger.info(f"Limited to first {args.story_limit} stories")
        
        logger.info(f"Loaded {len(stories)} stories")
        
        analyzer = MultiModelAnalyzer(config)
        analyzer.analyze_corpus(stories)
        
        logger.info("Analysis complete!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
