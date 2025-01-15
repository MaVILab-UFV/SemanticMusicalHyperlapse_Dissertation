#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(mixed_script_confusables, reason = "Mathematical notation.")]

mod accelerate;

use std::fs::File;
use std::io::{Cursor, Write};
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use ndarray::prelude::*;
use ndarray_npy::{NpzWriter, ReadNpyExt};

use crate::accelerate::{backward, forward_parallel, forward_sequential};

/// Runs the “Multimodal Frame Sampling Algorithm for Semantic Hyperlapses with Musical
/// Alignment”.
#[derive(Parser, Clone, Debug)]
#[clap(author, about, version)]
pub struct Opts {
    /// Path to the `.npy` file containing the video semantic features.
    #[clap(long)]
    semantic: PathBuf,
    /// Path to the `.npy` file containing the video transition matrix.
    #[clap(long)]
    matching: PathBuf,
    /// Path to the `.npy` file containing the audio features.
    #[clap(long)]
    alignment: PathBuf,
    /// Path to the `.json` file containing the weights of each optimization term.
    #[clap(long, default_value = "weights.json")]
    weights: PathBuf,
    /// Whether to use the parallel implementation.
    #[clap(long)]
    parallel: bool,
    /// The maximum momentary playback speed.
    #[clap(long, default_value_t = 20)]
    maximum_speed: usize,
    /// Path to a `.npz` file to save the results in.
    output: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let opts = Opts::parse();

    let weights: accelerate::Weights = {
        let reader = File::open(&opts.weights)
            .with_context(|| format!("Failed to open `{:?}`.", opts.weights))?;
        serde_json::from_reader(reader)
            .with_context(|| format!("Failed to read `{:?}`.", opts.weights))?
    };

    let semantic = {
        let reader = File::open(&opts.semantic)
            .with_context(|| format!("Failed to open `{:?}`.", opts.semantic))?;
        Array1::read_npy(reader)
            .with_context(|| format!("Failed to read `{:?}`.", opts.semantic))?
    };

    let alignment = {
        let reader = File::open(&opts.alignment)
            .with_context(|| format!("Failed to open `{:?}`.", opts.alignment))?;
        Array1::read_npy(reader)
            .with_context(|| format!("Failed to read `{:?}`.", opts.alignment))?
    };

    let matching = {
        let reader = File::open(&opts.matching)
            .with_context(|| format!("Failed to open `{:?}`.", opts.matching))?;
        Array2::read_npy(reader)
            .with_context(|| format!("Failed to read `{:?}`.", opts.matching))?
    };

    let forward = {
        let semantic = semantic.view();
        let matching = matching.view();
        let alignment = alignment.view();

        if opts.parallel {
            forward_parallel(
                semantic,
                matching,
                alignment,
                &weights,
                opts.maximum_speed,
            )
        } else {
            forward_sequential(
                semantic,
                matching,
                alignment,
                &weights,
                opts.maximum_speed,
            )
        }
    };
    let backward = backward(forward.view(), semantic.len(), alignment.len());

    // The intermediate buffer is needed when the path in `opts.save` is not seekable.
    let buf = {
        let mut buf = Cursor::new(Vec::new());
        let mut npz = NpzWriter::new(&mut buf);

        npz.add_array("index.npy", &backward.index.mapv(|n| n as u64))?;
        npz.add_array("speed.npy", &backward.speed.mapv(|n| n as u64))?;

        npz.add_array("audio.npy", &alignment)?;
        npz.add_array("video.npy", &semantic)?;

        npz.finish()?;
        buf.into_inner()
    };

    let mut writer = File::create(&opts.output)
        .with_context(|| format!("Failed to create `{:?}`.", opts.output))?;
    writer
        .write_all(&buf)
        .with_context(|| format!("Failed to write `{:?}`.", opts.output))?;

    Ok(())
}
