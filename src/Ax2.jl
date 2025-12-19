module Ax2

using WAV, DSP, LRUCache, Colors, Statistics, ImageCore, ImageMorphology, ImageFiltering, DelimitedFiles, HDF5

export load_recording, calculate_hanning_spectrograms, dB, overlay, precompute_configs, calculate_multitaper_spectrograms, coalesce_multitaper_power, coalesce_multitaper_ftest, make_strel, refine_ftest, scale_and_color, cumpower, play, get_components, save_csv, save_hdf
export freq, time, Periodograms
export app

fs_play = 48_000

load_recording(wavfile) = wavread(wavfile)

const Ys_cache = LRU{Tuple{UInt64,Int,Int,Int},DSP.Periodograms.Spectrogram}(maxsize=10)

"""
    calculate_hanning_spectrograms(y, nffts, noverlaps, fs, offset)

`y` has # time samples rows and # microphones columns.  `nffts` and `noverlaps`
are vectors of the same length.  `fs` is the sampling rate.  `offset` is just
used to debug.

returned is a matrix of spectrograms with # nffts rows and # mics columns
"""
function calculate_hanning_spectrograms(y, nffts, noverlaps, fs, offset)
    Ys = Matrix{DSP.Periodograms.Spectrogram}(undef, length(nffts), size(y,2))
    Threads.@threads :greedy for I in CartesianIndices((length(nffts), size(y,2)))
        infft, imic = I[1], I[2]
        Ys[infft,imic] = get!(Ys_cache, (hash(y[1+offset:end,imic]), nffts[infft], noverlaps[infft], offset)) do
            spectrogram(y[1+offset:end,imic], nffts[infft], noverlaps[infft]; fs=fs, window=hanning)
        end
    end
    return Ys
end

dB = x->20*log10.(x)
overlay(Ys::Matrix{<:DSP.Periodograms.Spectrogram}) = overlay(power.(Ys), dB)
overlay(Ys) = overlay(Ys, identity)
function overlay(Ys, fun)
    ntime, nfreq = size.(Ys,2), size.(Ys,1)
    mintime = minimum(round.(Int, maximum(ntime) ./ ntime) .* ntime)
    minfreq = minimum(round.(Int, maximum(nfreq) ./ nfreq) .* nfreq)
    Y_overlay = zeros(Float32, 4, minfreq, mintime, size(Ys,2))
    Threads.@threads :greedy for I in CartesianIndices(size(Ys))
        infft, imic = I[1], I[2]
        sz = size(Ys[infft,imic])
        scale = round.(Int, (minfreq, mintime) ./ sz)
        for f0 in 1:scale[1]
            fdelta = sz[1] - length(f0:scale[1]:minfreq)
            for t0 in 1:scale[2]
                tdelta = sz[2] - length(t0:scale[2]:mintime)
                Y_overlay[infft,
                  f0 : scale[1] : end,
                  t0 : scale[2] : end,
                  imic] .= fun.(@view Ys[infft,imic][1:end-fdelta, 1:end-tdelta])
            end
        end
    end
    Y_overlay[4,:,:,:] .= 1
    if size(Ys,1) == 2
        Y_overlay[3,:,:,:] .= @view Y_overlay[2,:,:,:]
        Y_overlay[2,:,:,:] .= @view Y_overlay[1,:,:,:]
    elseif size(Ys,1) == 1
        Y_overlay[2,:,:,:] .= Y_overlay[3,:,:,:] .= @view Y_overlay[1,:,:,:]
    end
    return Y_overlay
end

const configs = Dict{Int64,Channel{MTConfig{Float64}}}()

function precompute_config(nfft, nw, k, fs)
    configs[nfft] = Channel{MTConfig{Float64}}(Threads.nthreads())
    foreach(1:Threads.nthreads()) do _
        put!(configs[nfft], MTConfig{Float64}(nfft; ftest=true, nw=nw, ntapers=k, fs=fs))
    end
end

"""
    calculate_multitaper_spectrograms(y, nffts, noverlaps, nw, k, fs, iclip)

`y` has # time samples rows and # microphones columns.  `nffts` and `noverlaps`
are vectors of the same length.  `nw` and `k` are the time-bandwidth and number
of tapers, respectively.  `fs` is the sampling rate.  `iclip` is the portion of
time to analyse.

returned is a vector for each nfft, each element being a matrix of multi-taper
periodograms with # time slices rows and # mics columns
"""
function calculate_multitaper_spectrograms(y, nffts, noverlaps, nw, k, fs, iclip)
    function _mt_pgram(idx, imic, nfft, config)
        _y = y[idx:idx+nfft-1, imic]
        _y .-= mean(_y)
        mt_pgram(_y, config)
    end

    mtspectrums = Matrix{Periodograms.PeriodogramF}[]
    for (nfft,noverlap) in zip(nffts,noverlaps)
        haskey(configs, nfft) || precompute_config(nfft, nw, k, fs)
        idxs = iclip[1] : nfft-noverlap : iclip[2]+1-nfft+1
        mtspectrum = Matrix{Periodograms.PeriodogramF}(undef, length(idxs), size(y,2))
        i_idxs = Channel() do chnl
            foreach(i_idx->put!(chnl,i_idx), enumerate(idxs))
        end
        @sync for _ in 1:Threads.nthreads()
            Threads.@spawn begin
                config = take!(configs[nfft])
                for (i,idx) in i_idxs, imic in axes(y,2)
                    mtspectrum[i,imic] = _mt_pgram(idx, imic, nfft, config)
                end
                put!(configs[nfft], config)
            end
        end
        push!(mtspectrums, mtspectrum)
    end
    return mtspectrums
end

"""
returned is a matrix of multi-taper spectrograms with # nffts rows and # mics columns
"""
function coalesce_multitaper_power(mts)
    Y_MTs = Matrix{DSP.Periodograms.Spectrogram}(undef,
            length(mts), size(mts[1],2))
    for infft in 1:length(mts), imic in axes(mts[infft],2)
        p = hcat((power(x) for x in mts[infft][:,imic])...)
        Y_MTs[infft,imic] = DSP.Periodograms.Spectrogram(p, 0:0., 0:0.)
    end
    return Y_MTs
end

"""
returned is a matrix of multi-taper F-tests with # nffts rows and # mics columns
"""
function coalesce_multitaper_ftest(mts)
    Fs = Matrix{Matrix{Float64}}(undef, length(mts), size(mts[1],2))
    for infft in 1:length(mts), imic in axes(mts[infft],2)
        Fs[infft,imic] = reduce(hcat, Fpval.(mts[infft][:,imic]))
    end
    return Fs
end

make_strel(t) = strel_box(t)

function get_pad(pad, strel, F, val)
    if pad
        padlo = map(a->-first(a), axes(strel))
        padhi = last.(axes(strel))
        F_padded = padarray(view(F,1,:,:), Fill(val, padlo, padhi))
    else
        padlo = padhi = (0,0)
        F_padded = ImageCore.OffsetArray(F[1,:,:], 1:size(F,2), 1:size(F,3))
    end
    return F_padded, (1:1, 1+padlo[1]:size(F,2)+padlo[1], 1+padlo[2]:size(F,3)+padlo[2])
end

function refine_ftest(_Fs, minpix1, pval, anyall, sigonly,
                      morphclose, strelclose, morphopen, strelopen,
                      minpix2, pad)
    if sigonly && minpix1>0
        Fs = deepcopy(_Fs)
        Threads.@threads for thisFs in Fs
            labels = label_components(thisFs .< pval, trues(3,3))
            indices = component_indices(labels)
            for idx in Iterators.filter(x->length(x) < minpix1, Iterators.drop(indices, 1))
                thisFs[idx] .= 1
            end
        end
    else
        Fs = _Fs
    end
    F_overlay = overlay(Fs)
    Threads.@threads for imic in axes(Fs,2)
        for ifreq in axes(F_overlay,2)
            for itime in axes(F_overlay,3)
                if anyall(x->x<pval, view(F_overlay,1:3,ifreq,itime,imic))
                    F_overlay[:,ifreq,itime,imic] .= 1   # fuchsia
                    F_overlay[2,ifreq,itime,imic] = 0
                elseif sigonly
                    F_overlay[:,ifreq,itime,imic] .= 0   # black
                    F_overlay[4,ifreq,itime,imic] = 1
                end
            end
        end
        if sigonly
            if morphclose
                F_padded, ipad = get_pad(pad, strelclose, F_overlay[:,:,:,imic], 0)
                F_closed = closing(F_padded, strelclose)
                F_overlay[:,:,:,imic] .= reshape(F_closed, 1, size(F_closed)...)[ipad...]
                F_overlay[2,:,:,imic] .= 0
                F_overlay[4,:,:,imic] .= 1
            end
            if morphopen
                F_padded, ipad = get_pad(pad, strelopen, F_overlay[:,:,:,imic], 1)
                F_opened = opening(F_padded, strelopen)
                F_overlay[:,:,:,imic] .= reshape(F_opened, 1, size(F_opened)...)[ipad...]
                F_overlay[2,:,:,imic] .= 0
                F_overlay[4,:,:,imic] .= 1
            end
            if minpix2>0
                labels = label_components(F_overlay[1,:,:,imic], trues(3,3))
                indices = component_indices(labels)
                for idx in Iterators.filter(x->length(x) < minpix2, Iterators.drop(indices, 1))
                    view(F_overlay, 1,:,:,imic)[idx] .= 0
                    view(F_overlay, 2,:,:,imic)[idx] .= 0
                    view(F_overlay, 3,:,:,imic)[idx] .= 0
                end
            end
        end
    end
    return collect(dropdims(reinterpret(RGBA{N0f8}, N0f8.(F_overlay)), dims=1))
end

function scale_and_color(Y_overlay)
    q = quantile(Y_overlay[1:3,:,:], [0.01,0.99])
    f = scaleminmax(N0f8, q...)
    Y_scaled = f.(Y_overlay)
    Y_scaled[4,:,:] .= 1
    collect(dropdims(reinterpret(RGBA{N0f8}, Y_scaled), dims=1)')
end

function cumpower(d,p...)
    x = dropdims(sum(reduce(.+,p,init=p[1]*0.0), dims=d), dims=d)
    @. red(x) + green(x) + blue(x)
end

function play(y, iclip, fs)
    yfilt = filtfilt(digitalfilter(Lowpass(fs_play/2/fs[]), Butterworth(4)),
                     y[iclip[1]:iclip[2], 1])
    ydown = resample(yfilt, fs_play/fs)
    wavplay(ydown, fs_play)
end

function get_components(F, Y_freq, ifreq, Y_time, itime)
    labels = label_components(view(F,ifreq,itime), trues(3,3))
    indices = component_indices(CartesianIndex, labels)
    data = Matrix{Float64}(undef, length(indices)-2, 4)
    for (i,idx) in Iterators.drop(enumerate(indices), 2)
        bb = extrema(idx)
        lo, hi = Y_freq[ifreq][[bb[1].I[1], bb[2].I[1]]]
        start, stop = Y_time[itime][[bb[1].I[2], bb[2].I[2]]]
        data[i-2,:] .= (start, stop, lo, hi)
    end
    return data
end

function save_csv(filename, F, Y_freq, ifreq, Y_time, itime)
    data = get_components(F, Y_freq, ifreq, Y_time, itime)
    writedlm(filename, ["start (sec)" "stop (sec)" "low (Hz)" "high (Hz)"; data], ',')
end

function save_hdf(filename, F, Y_freq, ifreq, Y_time, itime)
    data = get_components(F, Y_freq, ifreq, Y_time, itime)
    fid = h5open(filename, "w")
    @write fid data
    HDF5.attributes(fid["data"])["header"] = "start (sec), stop (sec), low (Hz), high (Hz)"
    close(fid)
end

app() = nothing

end # module Ax2
