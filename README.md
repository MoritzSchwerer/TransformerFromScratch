# Implementation of transformer

This is an implementation of the transformer architecture from scratch.
I use pytorch to be able to run on the gpu but the transformer logic including the
self attention and multhead attention is implementet from scratch.

If you want to run and modify this you only need to install pytorch, nothing else.


### Example usage:
```python
# args: (batch_size, embed_size, seq_len, num_heads, forward_expand, drop_out_prob, max_seq_len)
model = Encoder(1000, 256, 10, 8, 4, 0.1, 50)
input = torch.randint(0, 1, (1000, 10))
output = model(input)
# input shape: torch.Size([1000, 10])
# output shape: torch.Size([1000, 10, 256])
# NOTE: you would typically only use the last embedding
# which would make the output shape: [1000, 256]
```

At this time I only implemented the encoder part of the transformer I will when I get
time also implement the decoder part of the architecture, which will not require much more effort
since the building blocks are in place.


