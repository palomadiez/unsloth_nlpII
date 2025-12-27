def format_dolly(examples, tokenizer):
    """Formatea y tokeniza ejemplos del dataset Dolly"""
    
    # Crear textos formateados
    texts = []
    for instruction, response in zip(examples["instruction"], examples["response"]):
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}{tokenizer.eos_token}"
        texts.append(text)
    
    # Tokenizar en lote
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding="max_length",  # Asegurar padding consistente
        return_tensors=None,   # Devolver listas en lugar de tensores
    )
    
    # Copiar input_ids a labels para language modeling
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized