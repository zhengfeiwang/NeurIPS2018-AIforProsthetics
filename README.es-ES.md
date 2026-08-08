

# Solución al Desafío de IA para Prótesis NeurIPS 2018

Una solución basada en OpenAI baselines ([repositorio](https://github.com/openai/baselines)) para el 9º puesto en el [Desafío de IA para Prótesis NeurIPS 2018](https://www.crowdai.org/challenges/neurips-2018-ai-for-prosthetics-challenge)

## Índice de Contenidos
* [Requisitos previos](#prerequisites)
* [Entrenamiento](#training)
* [Evaluación](#evaluation)
* [Contribución](#contributing)

## Requisitos previos

En primer lugar, debes seguir las instrucciones del entorno del desafío para configurar el simulador, los detalles se describen en: 
https://github.com/stanfordnmbl/osim-rl/  
```bash
source activate opensim-rl
```
Se requieren las siguientes dependencias adicionales:
- Ray
- TensorFlow
- mpi4py, cloudpickle, joblib (requeridas por baselines)

## Entrenamiento

Para entrenar el modelo desde cero, debes comentar `--checkpoint-path` en `run.sh`, o ejecutar los comandos a continuación:
```bash
python -m nips.round2_train \
    --num-timesteps=10000000 \
    --num-steps=128 \
    --num-minibatches=16 \
    --num-cpus=24 --num-casks=8 \
    --save-interval=5 \
    --repeat=2
```
El código base debe ser el mínimo necesario para este desafío, y seleccionamos PPO ([artículo](https://arxiv.org/abs/1707.06347)) como algoritmo.  
Debes especificar tus CPUs y GPUs disponibles en el script bash. El entrenamiento se ejecutará de forma distribuida mediante Ray ([documentación](https://ray.readthedocs.io/en/latest/index.html)).  

## Evaluación

Para evaluar tu modelo, guarda tus archivos de checkpoint en `checkpoints/` y modifica los argumentos en el script.  
```bash
# remember to place your checkpoints in your script file first
./run.sh
```
Para tener una referencia, nuestro modelo obtuvo finalmente 9813.224 en la Ronda 2 del desafío.

## Contribución

- [wangzhengfei0730](https://github.com/wangzhengfei0730)
- [QPHutu](https://github.com/QPHutu)
