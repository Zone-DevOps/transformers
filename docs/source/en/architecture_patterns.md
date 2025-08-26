<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Model Implementation Patterns

This document provides detailed examples of how the architectural patterns described in the [Library Architecture](./architecture) are implemented in practice, with concrete code flows and implementation details.

## Model Implementation Flow

This diagram shows the typical flow for implementing a new model in the transformers library, illustrating how the various architectural components work together.

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant Config as Configuration
    participant Model as Model Class
    participant AutoModel as AutoModel
    participant Registry as Model Registry
    participant Hub as Hugging Face Hub
    
    Dev->>Config: 1. Create ModelConfig class
    Config->>Config: Define architecture parameters
    
    Dev->>Model: 2. Implement Model class
    Model->>Model: Inherit from PreTrainedModel
    Model->>Model: Implement forward() method
    Model->>Model: Define model architecture
    
    Dev->>AutoModel: 3. Register with AutoModel
    AutoModel->>Registry: Add to model registry
    Registry->>Registry: Map config_class -> model_class
    
    Dev->>Hub: 4. Upload to Hub
    Hub->>Hub: Store model weights & config
    
    Note over Dev,Hub: Usage Flow
    
    participant User
    User->>AutoModel: AutoModel.from_pretrained("model_name")
    AutoModel->>Hub: Download config.json
    Hub->>AutoModel: Return configuration
    AutoModel->>Registry: Look up model class
    Registry->>AutoModel: Return model class
    AutoModel->>Model: Instantiate model
    AutoModel->>Hub: Download model weights
    Hub->>AutoModel: Return weights
    AutoModel->>Model: Load weights into model
    Model->>User: Return loaded model
```

## BERT Implementation Example

Let's trace through how BERT is implemented to understand the architecture patterns in practice.

```mermaid
classDiagram
    class BertConfig {
        +vocab_size: int = 30522
        +hidden_size: int = 768
        +num_hidden_layers: int = 12
        +num_attention_heads: int = 12
        +intermediate_size: int = 3072
        +max_position_embeddings: int = 512
        +type_vocab_size: int = 2
        +initializer_range: float = 0.02
        +layer_norm_eps: float = 1e-12
        +pad_token_id: int = 0
        +position_embedding_type: str = "absolute"
    }
    
    class BertEmbeddings {
        +word_embeddings: Embedding
        +position_embeddings: Embedding
        +token_type_embeddings: Embedding
        +LayerNorm: LayerNorm
        +dropout: Dropout
        +forward(input_ids, token_type_ids, position_ids)
    }
    
    class BertSelfAttention {
        +query: Linear
        +key: Linear
        +value: Linear
        +dropout: Dropout
        +num_attention_heads: int
        +attention_head_size: int
        +forward(hidden_states, attention_mask)
    }
    
    class BertSelfOutput {
        +dense: Linear
        +LayerNorm: LayerNorm
        +dropout: Dropout
        +forward(hidden_states, input_tensor)
    }
    
    class BertAttention {
        +self: BertSelfAttention
        +output: BertSelfOutput
        +forward(hidden_states, attention_mask)
    }
    
    class BertIntermediate {
        +dense: Linear
        +intermediate_act_fn: function
        +forward(hidden_states)
    }
    
    class BertOutput {
        +dense: Linear
        +LayerNorm: LayerNorm
        +dropout: Dropout
        +forward(hidden_states, input_tensor)
    }
    
    class BertLayer {
        +attention: BertAttention
        +intermediate: BertIntermediate
        +output: BertOutput
        +forward(hidden_states, attention_mask)
    }
    
    class BertEncoder {
        +layer: ModuleList[BertLayer]
        +forward(hidden_states, attention_mask)
    }
    
    class BertPooler {
        +dense: Linear
        +activation: Tanh
        +forward(hidden_states)
    }
    
    class BertModel {
        +config: BertConfig
        +embeddings: BertEmbeddings
        +encoder: BertEncoder
        +pooler: BertPooler
        +forward(input_ids, attention_mask, token_type_ids)
    }
    
    class BertForSequenceClassification {
        +bert: BertModel
        +dropout: Dropout
        +classifier: Linear
        +forward(input_ids, attention_mask, labels)
    }
    
    BertConfig --> BertModel
    BertModel *-- BertEmbeddings
    BertModel *-- BertEncoder
    BertModel *-- BertPooler
    BertEncoder *-- BertLayer
    BertLayer *-- BertAttention
    BertLayer *-- BertIntermediate
    BertLayer *-- BertOutput
    BertAttention *-- BertSelfAttention
    BertAttention *-- BertSelfOutput
    BertForSequenceClassification *-- BertModel
```

## Pipeline Implementation Deep Dive

This shows how a text classification pipeline processes data through the entire stack.

```mermaid
flowchart TD
    subgraph "Input Processing"
        INPUT[Raw Text: "This movie is great!"]
        PIPELINE[TextClassificationPipeline]
    end
    
    subgraph "Tokenization"
        TOKENIZER[BertTokenizer]
        TOKENS[Token IDs: [101, 2023, 3185, 2003, 2307, 999, 102]]
        ATTENTION[Attention Mask: [1, 1, 1, 1, 1, 1, 1]]
    end
    
    subgraph "Model Processing"
        EMBEDDINGS[Token Embeddings<br/>+ Position Embeddings<br/>+ Type Embeddings]
        ENCODER[BERT Encoder<br/>12 Transformer Layers]
        POOLER[Pooling Layer<br/>CLS Token Representation]
        CLASSIFIER[Classification Head<br/>Linear + Softmax]
    end
    
    subgraph "Output Processing"
        LOGITS[Raw Logits: [-1.2, 2.4]]
        PROBS[Probabilities: [0.15, 0.85]]
        LABELS[Labels: ["NEGATIVE", "POSITIVE"]]
        RESULT[{"label": "POSITIVE", "score": 0.85}]
    end
    
    INPUT --> PIPELINE
    PIPELINE --> TOKENIZER
    TOKENIZER --> TOKENS
    TOKENIZER --> ATTENTION
    
    TOKENS --> EMBEDDINGS
    ATTENTION --> EMBEDDINGS
    EMBEDDINGS --> ENCODER
    ENCODER --> POOLER
    POOLER --> CLASSIFIER
    
    CLASSIFIER --> LOGITS
    LOGITS --> PROBS
    PROBS --> LABELS
    LABELS --> RESULT
    
    PIPELINE --> RESULT
```

## Training Implementation Flow

This diagram shows how the Trainer orchestrates the training process with all the architectural components.

```mermaid
sequenceDiagram
    participant User
    participant Trainer
    participant DataLoader
    participant Model
    participant Optimizer
    participant Scheduler
    participant Callback
    participant Logger
    
    User->>Trainer: trainer.train()
    Trainer->>Callback: on_train_begin()
    
    loop For each epoch
        Trainer->>Callback: on_epoch_begin()
        Trainer->>DataLoader: Get training data
        
        loop For each batch
            DataLoader->>Trainer: Return batch
            Trainer->>Model: Forward pass
            Model->>Trainer: Return loss, logits
            
            Trainer->>Optimizer: Zero gradients
            Trainer->>Trainer: Backward pass (loss.backward())
            Trainer->>Optimizer: Step (update weights)
            Trainer->>Scheduler: Step (update learning rate)
            
            Trainer->>Callback: on_step_end()
            Trainer->>Logger: Log metrics
            
            alt Evaluation step
                Trainer->>Trainer: Run evaluation
                Trainer->>Callback: on_evaluate()
            end
            
            alt Save checkpoint
                Trainer->>Trainer: Save model state
                Trainer->>Callback: on_save()
            end
        end
        
        Trainer->>Callback: on_epoch_end()
    end
    
    Trainer->>Callback: on_train_end()
    Trainer->>User: Training complete
```

## Memory Optimization Implementation

This shows how various memory optimization techniques are applied during training and inference.

```mermaid
graph TB
    subgraph "Input Data"
        BATCH[Batch Size: 32<br/>Sequence Length: 512<br/>Memory: ~2GB]
    end
    
    subgraph "Gradient Checkpointing"
        FORWARD[Forward Pass<br/>Save only key activations]
        RECOMPUTE[Recompute activations<br/>during backward pass]
    end
    
    subgraph "Mixed Precision Training"
        FP16_FORWARD[FP16 Forward Pass<br/>Half memory usage]
        FP32_GRADIENTS[FP32 Gradient Updates<br/>Maintain precision]
        LOSS_SCALING[Loss Scaling<br/>Prevent underflow]
    end
    
    subgraph "Optimizer State Sharding"
        OPTIMIZER_SHARD[Shard optimizer states<br/>across devices]
        PARAM_SHARD[Shard model parameters<br/>across devices]
        GRADIENT_SHARD[Shard gradients<br/>across devices]
    end
    
    subgraph "CPU Offloading"
        CPU_OFFLOAD[Offload to CPU<br/>when not active]
        GPU_FETCH[Fetch to GPU<br/>when needed]
    end
    
    BATCH --> FORWARD
    FORWARD --> RECOMPUTE
    RECOMPUTE --> FP16_FORWARD
    FP16_FORWARD --> FP32_GRADIENTS
    FP32_GRADIENTS --> LOSS_SCALING
    LOSS_SCALING --> OPTIMIZER_SHARD
    OPTIMIZER_SHARD --> PARAM_SHARD
    PARAM_SHARD --> GRADIENT_SHARD
    GRADIENT_SHARD --> CPU_OFFLOAD
    CPU_OFFLOAD --> GPU_FETCH
```

## AutoModel Registration and Discovery

This shows the complete flow of how models are registered and discovered through the AutoModel system.

```mermaid
flowchart LR
    subgraph "Model Definition"
        CONFIG_DEF[Configuration Definition<br/>class MyModelConfig]
        MODEL_DEF[Model Definition<br/>class MyModel]
        TASK_DEF[Task-specific Models<br/>MyModelForClassification]
    end
    
    subgraph "Registration"
        AUTO_CONFIG[AutoConfig.register<br/>(MyModelConfig)]
        AUTO_MODEL[AutoModel.register<br/>(MyModelConfig, MyModel)]
        AUTO_TASK[AutoModelForSequenceClassification.register<br/>(MyModelConfig, MyModelForClassification)]
    end
    
    subgraph "Discovery Process"
        USER_REQUEST[AutoModel.from_pretrained<br/>("model_name")]
        CONFIG_LOAD[Load config.json]
        MODEL_TYPE[Extract model_type]
        REGISTRY_LOOKUP[Look up in registry]
        CLASS_INSTANTIATE[Instantiate model class]
        WEIGHTS_LOAD[Load pretrained weights]
    end
    
    subgraph "Runtime Registry"
        CONFIG_REGISTRY[CONFIG_MAPPING<br/>model_type -> config_class]
        MODEL_REGISTRY[MODEL_MAPPING<br/>config_class -> model_class]
        TASK_REGISTRY[MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING<br/>config_class -> task_model_class]
    end
    
    CONFIG_DEF --> AUTO_CONFIG
    MODEL_DEF --> AUTO_MODEL
    TASK_DEF --> AUTO_TASK
    
    AUTO_CONFIG --> CONFIG_REGISTRY
    AUTO_MODEL --> MODEL_REGISTRY
    AUTO_TASK --> TASK_REGISTRY
    
    USER_REQUEST --> CONFIG_LOAD
    CONFIG_LOAD --> MODEL_TYPE
    MODEL_TYPE --> REGISTRY_LOOKUP
    REGISTRY_LOOKUP --> CONFIG_REGISTRY
    REGISTRY_LOOKUP --> MODEL_REGISTRY
    REGISTRY_LOOKUP --> TASK_REGISTRY
    REGISTRY_LOOKUP --> CLASS_INSTANTIATE
    CLASS_INSTANTIATE --> WEIGHTS_LOAD
```

## Quantization Implementation Flow

This shows how different quantization strategies are implemented and applied to models.

```mermaid
graph TD
    subgraph "Model Loading"
        ORIGINAL[Original FP32 Model<br/>Memory: 100%]
        LOAD_CONFIG[Load Quantization Config]
    end
    
    subgraph "Quantization Strategies"
        DYNAMIC[Dynamic Quantization<br/>Runtime conversion]
        STATIC[Static Quantization<br/>Calibration dataset]
        QAT[Quantization Aware Training<br/>Train with fake quantization]
    end
    
    subgraph "Precision Options"
        FP16[Half Precision<br/>Memory: 50%]
        INT8[8-bit Quantization<br/>Memory: 25%]
        INT4[4-bit Quantization<br/>Memory: 12.5%]
    end
    
    subgraph "Implementation"
        BITSANDBYTES[BitsAndBytes<br/>GPU quantization]
        GGML[GGML/GGUF<br/>CPU quantization]
        OPENVINO[OpenVINO<br/>Intel optimization]
        ONNX_QUANT[ONNX Runtime<br/>Cross-platform]
    end
    
    subgraph "Performance Impact"
        SPEED[Inference Speed<br/>2-4x faster]
        MEMORY[Memory Usage<br/>50-88% reduction]
        ACCURACY[Accuracy Loss<br/>1-5% degradation]
    end
    
    ORIGINAL --> LOAD_CONFIG
    LOAD_CONFIG --> DYNAMIC
    LOAD_CONFIG --> STATIC
    LOAD_CONFIG --> QAT
    
    DYNAMIC --> FP16
    STATIC --> INT8
    QAT --> INT4
    
    FP16 --> BITSANDBYTES
    INT8 --> GGML
    INT4 --> OPENVINO
    STATIC --> ONNX_QUANT
    
    BITSANDBYTES --> SPEED
    GGML --> MEMORY
    OPENVINO --> ACCURACY
    ONNX_QUANT --> SPEED
```

## Generation Strategy Implementation

This shows how different text generation strategies are implemented and orchestrated.

```mermaid
stateDiagram-v2
    [*] --> TokenizeInput
    
    state "Input Processing" as InputProc {
        TokenizeInput --> EncodePrompt
        EncodePrompt --> InitializeGeneration
    }
    
    state "Generation Strategy Selection" as StratSelect {
        InitializeGeneration --> GreedyDecoding
        InitializeGeneration --> BeamSearch
        InitializeGeneration --> Sampling
        InitializeGeneration --> ContrastiveSearch
    }
    
    state "Greedy Decoding" as GreedyDecoding {
        GreedyDecoding --> SelectBestToken
        SelectBestToken --> AppendToken
    }
    
    state "Beam Search" as BeamSearch {
        BeamSearch --> MaintainBeams
        MaintainBeams --> ScoreSequences
        ScoreSequences --> PruneBeams
    }
    
    state "Sampling Methods" as Sampling {
        Sampling --> TopKSampling
        Sampling --> TopPSampling
        Sampling --> TemperatureScaling
        TopKSampling --> SampleToken
        TopPSampling --> SampleToken
        TemperatureScaling --> SampleToken
    }
    
    state "Contrastive Search" as ContrastiveSearch {
        ContrastiveSearch --> ComputeSimilarity
        ComputeSimilarity --> BalanceConfidenceDegeneration
    }
    
    state "Constraint Checking" as ConstraintCheck {
        AppendToken --> ConstraintCheck
        PruneBeams --> ConstraintCheck
        SampleToken --> ConstraintCheck
        BalanceConfidenceDegeneration --> ConstraintCheck
        
        ConstraintCheck --> CheckStoppingCriteria
        CheckStoppingCriteria --> CheckMaxLength
        CheckMaxLength --> CheckEOSToken
        CheckEOSToken --> ApplyRepetitionPenalty
    }
    
    state "Termination" as Termination {
        ApplyRepetitionPenalty --> Continue: continue
        ApplyRepetitionPenalty --> DecodeTokens: stop
        DecodeTokens --> ReturnGeneratedText
    }
    
    Continue --> StratSelect
    ReturnGeneratedText --> [*]
```

## Backend Integration Implementation

This shows how the library maintains compatibility across different ML frameworks.

```mermaid
graph TB
    subgraph "Common Interface Layer"
        COMMON_CONFIG[Shared Configuration Classes]
        COMMON_UTILS[Shared Utilities]
        COMMON_HUB[Hub Integration]
    end
    
    subgraph "PyTorch Implementation"
        TORCH_MODEL[PyTorch Models<br/>modeling_*.py]
        TORCH_LAYERS[PyTorch Layers<br/>nn.Module subclasses]
        TORCH_UTILS[PyTorch Utilities<br/>torch specific functions]
        TORCH_TRAIN[PyTorch Training<br/>Trainer class]
    end
    
    subgraph "TensorFlow Implementation"
        TF_MODEL[TensorFlow Models<br/>modeling_tf_*.py]
        TF_LAYERS[TensorFlow Layers<br/>tf.keras.layers subclasses]
        TF_UTILS[TensorFlow Utilities<br/>tf specific functions]
        TF_TRAIN[TensorFlow Training<br/>TFTrainer class]
    end
    
    subgraph "JAX/Flax Implementation"
        FLAX_MODEL[Flax Models<br/>modeling_flax_*.py]
        FLAX_LAYERS[Flax Layers<br/>flax.linen.Module subclasses]
        FLAX_UTILS[Flax Utilities<br/>jax specific functions]
        FLAX_TRAIN[Flax Training<br/>FlaxTrainer class]
    end
    
    subgraph "Cross-Framework Features"
        WEIGHT_CONVERSION[Weight Conversion<br/>PyTorch ↔ TensorFlow ↔ Flax]
        CONSISTENT_API[Consistent APIs<br/>Same method signatures]
        SHARED_CONFIGS[Shared Configurations<br/>Framework-agnostic parameters]
    end
    
    COMMON_CONFIG --> TORCH_MODEL
    COMMON_CONFIG --> TF_MODEL
    COMMON_CONFIG --> FLAX_MODEL
    
    COMMON_UTILS --> TORCH_UTILS
    COMMON_UTILS --> TF_UTILS
    COMMON_UTILS --> FLAX_UTILS
    
    COMMON_HUB --> TORCH_TRAIN
    COMMON_HUB --> TF_TRAIN
    COMMON_HUB --> FLAX_TRAIN
    
    TORCH_MODEL --> WEIGHT_CONVERSION
    TF_MODEL --> WEIGHT_CONVERSION
    FLAX_MODEL --> WEIGHT_CONVERSION
    
    TORCH_UTILS --> CONSISTENT_API
    TF_UTILS --> CONSISTENT_API
    FLAX_UTILS --> CONSISTENT_API
    
    TORCH_TRAIN --> SHARED_CONFIGS
    TF_TRAIN --> SHARED_CONFIGS
    FLAX_TRAIN --> SHARED_CONFIGS
```

## Summary

These implementation patterns demonstrate the key principles of the transformers library architecture:

1. **Consistent Abstractions**: All models follow the same base class patterns and interfaces
2. **Modular Design**: Components can be mixed and matched (different tokenizers with different models)
3. **Framework Agnostic**: Same high-level APIs work across PyTorch, TensorFlow, and JAX/Flax
4. **Extensible Registration**: New models and configurations can be easily added to the AutoModel system
5. **Performance Optimization**: Multiple strategies for memory and compute optimization
6. **User-Friendly APIs**: High-level pipelines abstract away complexity while low-level APIs provide control

These patterns enable the library to maintain backwards compatibility while continuously adding new models and features, making it both powerful for researchers and accessible for practitioners.