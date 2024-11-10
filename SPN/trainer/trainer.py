import os
import json
import time

import torch, random, gc
from torch import nn, optim
from tqdm import tqdm
from transformers import AdamW
from utils.average_meter import AverageMeter
from utils.functions import formulate_gold, Char_Token_SpanConverter
from utils.metric import metric, num_metric, overlap_metric


def trans_to_string_case(pred_in, label_file, tokenizer, rela_list):
    """
    将 pred 由数值形式转换为自然语言形式
    pred_in:
        {
            "0": [
                [
                    0, 0.45815354585647583,      # relation
                    1, 5, 1.0, 0.9999983310699463,   # subj
                    8, 9, 0.9999336004257202, 0.99913090467453  # obj
                ],                                   # one triple
                [
                    3, 0.9994286894798279,
                    1, 5, 1.0, 0.9999864101409912,
                    11, 18, 0.9999997615814209, 1.0
                ],
                ...
            ],                                      # one sample
            ...
        }
    rela_list: self.data
    """

    # 输入的 pred_in 并不是注释中的 dict，目前只能通过json转换复现出来
    with open("temp.json", "w", encoding="utf-8") as fp:
        json.dump(pred_in, fp, ensure_ascii=False, indent=4)
    with open("temp.json", "r", encoding="utf-8") as fp:
        pred_in = json.loads(fp.read())

    span_converter = Char_Token_SpanConverter(
        tokenizer, add_special_tokens=True, has_return_offsets_mapping=False)

    with open(label_file, 'r', encoding='UTF-8') as f:
        label_data = json.loads(f.read())

    preds_out = []
    for d_i in range(len(label_data)):

        text = label_data[d_i]['text']

        triple_in_list = pred_in[str(d_i)]
        triples_out = []
        for triple_in in triple_in_list:
            rela = rela_list[triple_in[0]]
            subj_tok_span = (triple_in[2], triple_in[3]+1)
            obj_tok_span = (triple_in[6], triple_in[7]+1)
            subj_char_span = span_converter.get_char_span(text, subj_tok_span)
            obj_char_span = span_converter.get_char_span(text, obj_tok_span)
            subj = text[subj_char_span[0]:subj_char_span[1]]
            obj = text[obj_char_span[0]:obj_char_span[1]]
            triples_out.append({
                "subject": subj,
                "predicate": rela,
                "object": obj,
                "subj_char_span": subj_char_span,
                "obj_char_span": obj_char_span,
            })

        preds_out.append({
            "id": d_i,
            "text": text,
            "triple_pred_list": triples_out.copy(),
        })

    # print(f"-- pred example:\n  {preds_out[0]}")
    # time.sleep(5)
    return preds_out


class Trainer(nn.Module):
    def __init__(self, model, data, args, tokenizer):
        super().__init__()
        self.args = args
        self.model = model
        self.data = data
        self.relation_list = data.relational_alphabet.instances
        self.tokenizer = tokenizer   # 用于输出预测结果时，将结果转化为自然语言形式
        print(f"class Trainer - def __init__()\n  self.relation_list={self.relation_list}")
        time.sleep(5)

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        component = ['encoder', 'decoder']
        grouped_params = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and component[0] in n],
                'weight_decay': args.weight_decay,
                'lr': args.encoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and component[0] in n],
                'weight_decay': 0.0,
                'lr': args.encoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if
                           not any(nd in n for nd in no_decay) and component[1] in n],
                'weight_decay': args.weight_decay,
                'lr': args.decoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if
                           any(nd in n for nd in no_decay) and component[1] in n],
                'weight_decay': 0.0,
                'lr': args.decoder_lr
            }
        ]
        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(grouped_params)
        elif args.optimizer == 'AdamW':
            self.optimizer = AdamW(grouped_params)
        else:
            raise Exception("Invalid optimizer.")
        if args.use_gpu:
            self.cuda()

        if os.path.exists(self.args.output_dir) == 0:
            os.makedirs(self.args.output_dir)

    def train_model(self):
        best_f1 = 0
        best_result_epoch = -1
        train_loader = self.data.train_loader
        train_num = len(train_loader)
        batch_size = self.args.batch_size
        total_batch = train_num // batch_size + 1
        # result = self.eval_model(self.data.test_loader)
        for epoch in range(self.args.max_epoch):
            # Train
            self.model.train()
            self.model.zero_grad()
            self.optimizer = self.lr_decay(self.optimizer, epoch, self.args.lr_decay)
            print("=== Epoch %d train ===" % epoch, flush=True)
            avg_loss = AverageMeter()
            random.shuffle(train_loader)
            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > train_num:
                    end = train_num
                train_instance = train_loader[start:end]
                # print([ele[0] for ele in train_instance])
                if not train_instance:
                    continue
                input_ids, attention_mask, targets, _ = self.model.batchify(train_instance)
                loss, _ = self.model(input_ids, attention_mask, targets)
                avg_loss.update(loss.item(), 1)
                # Optimize
                loss.backward()
                if self.args.max_grad_norm != 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                if (batch_id + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.model.zero_grad()
                if batch_id % 100 == 0 and batch_id != 0:
                    print("     Instance: %d; loss: %.4f" % (start, avg_loss.avg), flush=True)

            gc.collect()
            torch.cuda.empty_cache()

            # Validation
            print("=== Epoch %d Validation ===" % epoch)
            metric_dev, res_dev = self.eval_model(self.data.valid_loader)
            res_dev = trans_to_string_case(pred_in=res_dev, label_file=self.args.valid_file,
                tokenizer=self.tokenizer, rela_list=self.relation_list,)

            # Test
            print("=== Epoch %d Test ===" % epoch, flush=True)
            metric_test, res_test = self.eval_model(self.data.test_loader)
            res_test = trans_to_string_case(pred_in=res_test, label_file=self.args.test_file,
                tokenizer=self.tokenizer, rela_list=self.relation_list,)

            # save
            output_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch{epoch}")
            if os.path.exists(output_dir) == 0:
                os.makedirs(output_dir)
            # torch.save(self.model.state_dict(), os.path.join(output_dir, "model_params.pt"))  # save parameters of model
            with open(os.path.join(output_dir, "dataset_prediction_dev.json"), "w", encoding="utf-8") as fp:
                json.dump(res_dev, fp, ensure_ascii=False, indent=4)
            with open(os.path.join(output_dir, "dataset_prediction_test.json"), "w", encoding="utf-8") as fp:
                json.dump(res_test, fp, ensure_ascii=False, indent=4)

            f1 = metric_dev['f1']
            if f1 > best_f1:
                print("Achieving Best Result on Validation Set.", flush=True)
                # torch.save(
                #     {'state_dict': self.model.state_dict()},
                #     self.args.generated_param_directory + " %s_%s_epoch_%d_f1_%.4f.model" %(self.model.name, self.args.dataset_name, epoch, result['f1'])
                # )
                best_f1 = f1
                best_result_epoch = epoch
            # if f1 <= 0.3 and epoch >= 10:
            #     break
            gc.collect()
            torch.cuda.empty_cache()
        print("Best result on validation set is %f achieving at epoch %d." % (best_f1, best_result_epoch), flush=True)

    def eval_model(self, eval_loader):
        self.model.eval()
        # print(self.model.decoder.query_embed.weight)
        prediction, gold = {}, {}
        with torch.no_grad():
            batch_size = self.args.batch_size
            eval_num = len(eval_loader)
            total_batch = eval_num // batch_size + 1

            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > eval_num:
                    end = eval_num
                eval_instance = eval_loader[start:end]
                if not eval_instance:
                    continue
                input_ids, attention_mask, target, info = self.model.batchify(eval_instance)
                gold.update(formulate_gold(target, info))
                # print(target)
                gen_triples = self.model.gen_triples(input_ids, attention_mask, info)
                prediction.update(gen_triples)
        num_metric(prediction, gold)
        overlap_metric(prediction, gold)
        # res = {"gold": gold, "pred": prediction, }   # 所有案例返回
        res = prediction   # 所有案例返回
        return metric(prediction, gold), res

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    @staticmethod
    def lr_decay(optimizer, epoch, decay_rate):
        # lr = init_lr * ((1 - decay_rate) ** epoch)
        if epoch != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * (1 - decay_rate)
                # print(param_group['lr'])
        return optimizer
