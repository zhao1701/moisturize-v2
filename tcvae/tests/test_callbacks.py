from tcvae.tests.fixtures import (
    datagen, model, reconstruction_check, traversal_check)


def test_reconstruction_check(model, datagen, reconstruction_check):
    callbacks = [reconstruction_check]
    model.compile('adam')
    model.fit_generator(datagen, epochs=2, callbacks=callbacks)
    

def test_traversal_check(model, datagen, traversal_check):
    callbacks = [traversal_check]
    model.compile('adam')
    model.fit_generator(datagen, epochs=2, callbacks=callbacks)

