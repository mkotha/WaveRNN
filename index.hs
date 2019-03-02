#!/usr/bin/env stack
-- stack --resolver=lts-13.9 runghc --package blaze-html
{-# LANGUAGE OverloadedStrings #-}

import qualified Data.Text as T
import Text.Blaze.Html5((!))
import qualified Text.Blaze.Html5 as H
import qualified Text.Blaze.Html5.Attributes as A
import qualified Text.Blaze.Html.Renderer.Pretty as Pretty

main :: IO ()
main = putStrLn $ Pretty.renderHtml page

page :: H.Html
page = H.docTypeHtml $ do
  H.head $
    H.title "Audio samples from mkotha/WaveRNN"
  H.body $ do
    H.h1 $ do
      "Audio samples from"
      H.a ! A.href "https://github.com/mkotha/WaveRNN" $ "mkotha/WaveRNN"

    H.h2 "VQ-VAE (VCTK dataset)"
    vqvaeSample "vq.42.vctk/945k_steps_0"
    vqvaeSample "vq.42.vctk/945k_steps_1"

    H.h2 "Vocoder (LJ Speech dataset)"
    vocoderSample "wavernn.43.upconv/1135k_steps_0"
    vocoderSample "wavernn.43.upconv/1135k_steps_1"
    vocoderSample "wavernn.43.upconv/1135k_steps_2"

vqvaeSample :: T.Text -> H.Html
vqvaeSample name = H.p $ do
  "original: "
  audioSample $ name <> "_target"
  ", reconstruction: "
  audioSample $ name <> "_generated"
  ", converted: "
  audioSample $ name <> "_transferred"
  "."

vocoderSample :: T.Text -> H.Html
vocoderSample name = H.p $ do
  "original: "
  audioSample $ name <> "_target"
  ", generated: "
  audioSample $ name <> "_generated"
  "."

audioSample :: T.Text -> H.Html
audioSample name = H.audio ! A.controls "controls" $ do
  H.source ! A.src (H.textValue wavPath) ! A.type_ "audio/wav"
  H.a ! A.href (H.textValue wavPath) $ H.text name
  where
    wavPath = "audio/" <> name <> ".wav"
