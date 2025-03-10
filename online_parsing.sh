cd benchmark/evaluation
python LILAC_eval.py -full --shot 1 --example_size 1 --model gemini-1.5-flash

# for a in 8 16 32 64 128
# do
#   for b in 3 8
#   do
#     python LLMTree_eval.py -full --shot $a --example_size $b --model gpt-3.5-turbo-0613
#     rm -rf ../../temp/*
#   done
# done

