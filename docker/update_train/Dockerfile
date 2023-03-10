FROM eric11eca/meta_kg_8:latest

ARG DUMMY=unknown
RUN DUMMY=${DUMMY}
RUN mkdir -p /nlp
RUN mkdir -p /nlp/output/
RUN mkdir -p /nlp/tensor/
COPY meta_kg/ /nlp/meta_kg/
COPY run_maml.py /nlp/
COPY cli_maml.py /nlp/
COPY run_eval.sh /nlp/
COPY run_gpt2.sh /nlp/

ENV WANDB_API_KEY 9edee5b624841e10c88fcf161d8dc54d4efbee29

ENTRYPOINT ["./run_gpt2.sh"]