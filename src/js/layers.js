import * as tf from '@tensorflow/tfjs';

class Sub extends tf.layers.Layer {
    constructor() {
        super({});
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = inputs;
            let output = inputs[0].clone();
            return tf.sub(output, inputs[1]);
        });
    }

    computeOutputShape(inputShape) {
        return inputShape[0];
    }
}
Sub.className = 'Sub';
export { Sub };
tf.serialization.registerClass(Sub);

class Mean extends tf.layers.Layer {
    constructor(axis) {
        super({});
        this.axis = axis;
    }

    call(input, kwargs) {
        return tf.tidy(() => {
            input = input;
            return tf.mean(input, this.axis, true);
        });
    }

    computeOutputShape(inputShape) {
        return inputShape;
    }
}
Mean.className = 'Mean';
export { Mean };
tf.serialization.registerClass(Mean);