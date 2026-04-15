import torch


class MaskingDependencies:
    def __init__(self, model, functions_schema):
        self.model = model

        self.token_lbrace = self._get_id("{")
        self.token_rbrace = self._get_id("}")
        self.token_quote = self._get_id('"')
        self.token_colon = self._get_id(":")
        self.token_comma = self._get_id(",")

        self.name_key_tokens = self.model.encode('"name"')[0].tolist()
        self.params_key_tokens = self.model.encode('"parameters"')[0].tolist()

        self.function_tokens = {
            f['name']: self.model.encode(f'"{f["name"]}"')[0].tolist()
            for f in functions_schema
        }

    def _mask_logits(self, logits, allowed_token_ids):
        mask = torch.full_like(logits, float('-inf'))
        for tid in allowed_token_ids:
            mask[tid] = logits[tid]
        return mask

    def _get_id(self, text):
        return self.model.encode(text)[0][-1].item()

    def apply_constrain(self, logits, current_ids):
        text_so_far = self.model.decode(current_ids)
        logits_tensor = torch.tensor(logits)

        if "Result: " not in text_so_far:
            return logits_tensor
        current_gen = text_so_far.split("Result: ")[-1]

        if current_gen.endswith('"name": "'):
            allowed = []
            for name in self.function_tokens.keys():
                allowed.append(self.model.encode(name)[0][0].item())
            return self._mask_logits(logits_tensor, allowed)

        if current_gen.count('"') == 4 and not current_gen.endswith(','):
            return self._mask_logits(logits_tensor, [self.token_comma])

        if current_gen.count('{') == current_gen.count('}'):
            if current_gen.count('{') > 0:
                return self._mask_logits(logits_tensor, [self.token_rbrace])

        return logits_tensor
