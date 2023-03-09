def build_model(model_class, args):
    config_class, model_class, tokenizer_class = model_class
    # 导入分类数
    config = config_class.from_pretrained(
        args.model_name_or_path
    )
    # bert
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=False,
    )
    # tokenizer = tokenizer_class.from_pretrained(
    #     args.model_name_or_path
    # )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
    )
    return tokenizer, tokenizer, model