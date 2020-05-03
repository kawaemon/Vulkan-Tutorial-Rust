#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]

#[macro_use]
extern crate vulkano;
extern crate vulkano_shaders;
extern crate vulkano_win;
extern crate winit;

use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use vulkano_win::VkSurfaceBuild;

use vulkano::{
    command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState},
    descriptor::PipelineLayoutAbstract,
    device::{Device, DeviceExtensions, Features, Queue},
    format::Format,
    framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass},
    image::{swapchain::SwapchainImage, ImageUsage},
    instance::{
        debug::{DebugCallback, MessageSeverity, MessageType},
        layers_list, ApplicationInfo, Instance, PhysicalDevice, Version,
    },
    pipeline::{
        vertex::{BufferlessDefinition, BufferlessVertices},
        viewport::Viewport,
        GraphicsPipeline,
    },
    swapchain::{
        acquire_next_image, AcquireError, Capabilities, ColorSpace, CompositeAlpha,
        FullscreenExclusive, PresentMode, SupportedPresentModes, Surface, Swapchain,
    },
    sync::{self, FlushError, GpuFuture, SharingMode},
};

use std::collections::HashSet;
use std::iter::FromIterator;
use std::sync::Arc;

const ValidationLayers: &[&str] = &["VK_LAYER_KHRONOS_validation"];
const Title: &'static str = "Vulkan Playground With Rust";
const Width: i32 = 800;
const Height: i32 = 600;

type ConcrateGraphicsPipeline = GraphicsPipeline<
    BufferlessDefinition,
    Box<PipelineLayoutAbstract + Send + Sync + 'static>,
    Arc<RenderPassAbstract + Send + Sync + 'static>,
>;

mod VertexShader {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450
#extension GL_ARB_separate_shader_objects: enable

out gl_PerVertex {
    vec4 gl_Position;
};

layout(location = 0) out vec3 fragColor;

vec2 positions[3] = vec2[] (
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

vec3 colors[3] = vec3[] (
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    fragColor = colors[gl_VertexIndex];
}"
    }
}

mod FragmentShader {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450
#extension GL_ARB_separate_shader_objects: enable

layout(location = 0) in vec3  fragColor;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}"
    }
}

fn RequiredDeviceExtensions() -> DeviceExtensions {
    DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    }
}

struct QueueFamilyIndices {
    GraphicsFamily: Option<usize>,
    PresentFamily: Option<usize>,
}

impl QueueFamilyIndices {
    fn new() -> QueueFamilyIndices {
        QueueFamilyIndices {
            GraphicsFamily: None,
            PresentFamily: None,
        }
    }

    fn isComplete(&self) -> bool {
        self.GraphicsFamily.is_some() && self.PresentFamily.is_some()
    }
}

struct App {
    EventLoop: EventLoop<()>,
    Surface: Arc<Surface<Window>>,
    Instance: Arc<Instance>,
    DebugCallback: DebugCallback,
    PhysicalDeviceIndex: usize,
    Device: Arc<Device>,
    GraphicsQueue: Arc<Queue>,
    PresentQueue: Arc<Queue>,
    Swapchain: Arc<Swapchain<Window>>,
    SwapchainImages: Vec<Arc<SwapchainImage<Window>>>,
    RenderPass: Arc<RenderPassAbstract + Send + Sync>,
    GraphicsPipeline: Arc<ConcrateGraphicsPipeline>,
    SwapchainFramebuffers: Vec<Arc<FramebufferAbstract + Send + Sync>>,
    CommandBuffers: Vec<Arc<AutoCommandBuffer>>,
}

impl App {
    fn CheckValidationLayerSupport() -> bool {
        let SupportedLayers = layers_list()
            .unwrap()
            .map(|layer| String::from(layer.name()))
            .collect::<Vec<String>>();

        println!("SupportedLayers: {:?}", SupportedLayers);
        println!("NeededLayers: {:?}", ValidationLayers);

        ValidationLayers
            .iter()
            .all(|name| SupportedLayers.contains(&String::from(*name)))
    }

    fn SetupDebugCallback(Instance: &Arc<Instance>) -> DebugCallback {
        let Severity = MessageSeverity::errors_and_warnings();
        let Types = MessageType::all();

        DebugCallback::new(Instance, Severity, Types, |Message| {
            println!("Validation layer message: {:?}", Message.description);
        })
        .unwrap()
    }

    fn PickPhysicalDevice(Instance: &Arc<Instance>, Surface: &Arc<Surface<Window>>) -> usize {
        PhysicalDevice::enumerate(Instance)
            .position(|Device| Self::isDeviceSuitable(Surface, &Device))
            .expect("Not found a suitable Physical Device")
    }

    fn CheckDeviceExtensionSupport(Device: &PhysicalDevice) -> bool {
        let AvailableExtensions = DeviceExtensions::supported_by_device(*Device);
        let RequiredExtensions = RequiredDeviceExtensions();

        AvailableExtensions.intersection(&RequiredExtensions) == RequiredExtensions
    }

    fn isDeviceSuitable(Surface: &Arc<Surface<Window>>, Device: &PhysicalDevice) -> bool {
        let Indices = Self::FindQueueFamilies(Device, Surface);
        let ExtensionsSupported = Self::CheckDeviceExtensionSupport(Device);

        let SwapChainAdaquate = if ExtensionsSupported {
            let Capabilities = Surface
                .capabilities(*Device)
                .expect("Failed to get surface capabilities");
            !Capabilities.supported_formats.is_empty()
                && Capabilities.present_modes.iter().next().is_some()
        } else {
            false
        };

        Indices.isComplete() && ExtensionsSupported && SwapChainAdaquate
    }

    fn FindQueueFamilies(
        Device: &PhysicalDevice,
        Surface: &Arc<Surface<Window>>,
    ) -> QueueFamilyIndices {
        let mut Indices = QueueFamilyIndices::new();
        for (ID, QueueFamily) in Device.queue_families().enumerate() {
            if QueueFamily.supports_graphics() {
                Indices.GraphicsFamily = Some(ID);
            }

            if Surface.is_supported(QueueFamily).unwrap() {
                Indices.PresentFamily = Some(ID);
            }

            if Indices.isComplete() {
                break;
            }
        }
        Indices
    }

    fn ChooseSwapSurfaceFormat(AvailableFormats: &[(Format, ColorSpace)]) -> (Format, ColorSpace) {
        *AvailableFormats
            .iter()
            .find(|(Format, ColorSpace)| {
                *Format == Format::B8G8R8A8Unorm && *ColorSpace == ColorSpace::SrgbNonLinear
            })
            .unwrap_or_else(|| &AvailableFormats[0])
    }

    fn ChooseSwapPresentMode(AvailablePresentModes: SupportedPresentModes) -> PresentMode {
        if AvailablePresentModes.mailbox {
            return PresentMode::Mailbox;
        }

        if AvailablePresentModes.immediate {
            return PresentMode::Immediate;
        }

        PresentMode::Fifo
    }

    fn ChooseSwapExtent(Capabilities: &Capabilities) -> [u32; 2] {
        if let Some(CurrentExtent) = Capabilities.current_extent {
            return CurrentExtent;
        } else {
            let mut ActualExtent = [Width as u32, Height as u32];
            for i in 0..2 {
                ActualExtent[i] = Capabilities.min_image_extent[i]
                    .max(Capabilities.max_image_extent[i].min(ActualExtent[i]))
            }

            return ActualExtent;
        }
    }

    fn CreateVulkanInstance() -> Arc<Instance> {
        let AppInfo = ApplicationInfo {
            application_name: Some(Title.into()),
            application_version: Some(Version {
                major: 0,
                minor: 0,
                patch: 0,
            }),
            engine_name: Some("No Engine".into()),
            engine_version: Some(Version {
                major: 0,
                minor: 0,
                patch: 0,
            }),
        };

        let mut RequiredExtension = vulkano_win::required_extensions();
        RequiredExtension.ext_debug_utils = true;

        return if Self::CheckValidationLayerSupport() {
            Instance::new(
                Some(&AppInfo),
                &RequiredExtension,
                ValidationLayers.iter().cloned(),
            )
            .unwrap()
        } else {
            println!("Validation not supported.");
            Instance::new(Some(&AppInfo), &RequiredExtension, None).unwrap()
        };
    }

    fn CreateLogicalDevice(
        Instance: &Arc<Instance>,
        Surface: &Arc<Surface<Window>>,
        PhysicalDeviceIndex: usize,
    ) -> (Arc<Device>, Arc<Queue>, Arc<Queue>) {
        let PhysicalDevice = PhysicalDevice::from_index(Instance, PhysicalDeviceIndex).unwrap();
        let Indices = Self::FindQueueFamilies(&PhysicalDevice, Surface);

        let Families = [
            Indices.GraphicsFamily.unwrap() as i32,
            Indices.PresentFamily.unwrap() as i32,
        ];
        let UniqueQueueFamilies: HashSet<&i32> = HashSet::from_iter(Families.iter());

        let QueuePriority = 1.0;
        let QueueFamilies = UniqueQueueFamilies.iter().map(|i| {
            (
                PhysicalDevice.queue_families().nth(**i as usize).unwrap(),
                QueuePriority,
            )
        });

        let (Device, mut Queues) = Device::new(
            PhysicalDevice,
            &Features::none(),
            &RequiredDeviceExtensions(),
            QueueFamilies,
        )
        .expect("Failed to create logical device.");

        let GraphicsQueue = Queues.next().unwrap();
        let PresentQueue = Queues.next().unwrap_or_else(|| GraphicsQueue.clone());

        (Device, GraphicsQueue, PresentQueue)
    }

    fn CreateSwapChain(
        Instance: &Arc<Instance>,
        Surface: &Arc<Surface<Window>>,
        PhysicalDeviceIndex: usize,
        Device: &Arc<Device>,
        GraphicsQueue: &Arc<Queue>,
        PresentQueue: &Arc<Queue>,
    ) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
        let PhysicalDevice = PhysicalDevice::from_index(Instance, PhysicalDeviceIndex).unwrap();
        let Capabilities = Surface
            .capabilities(PhysicalDevice)
            .expect("Failed to get surface capabilities");

        let SurfaceFormat = Self::ChooseSwapSurfaceFormat(&Capabilities.supported_formats);
        let PresentMode = Self::ChooseSwapPresentMode(Capabilities.present_modes);
        let Extent = Self::ChooseSwapExtent(&Capabilities);

        let mut ImageCount = Capabilities.min_image_count + 1;
        if let Some(MaxImageCount) = Capabilities.max_image_count {
            if ImageCount > MaxImageCount {
                ImageCount = MaxImageCount;
            }
        }

        let ImageUsage = ImageUsage {
            color_attachment: true,
            ..ImageUsage::none()
        };

        let Indices = Self::FindQueueFamilies(&PhysicalDevice, Surface);

        let SharingMode: SharingMode = if Indices.GraphicsFamily != Indices.PresentFamily {
            vec![GraphicsQueue, PresentQueue].as_slice().into()
        } else {
            GraphicsQueue.into()
        };

        let (SwapChain, Images) = Swapchain::new(
            Device.clone(),
            Surface.clone(),
            ImageCount,
            SurfaceFormat.0,
            Extent,
            1, //Layers
            ImageUsage,
            SharingMode,
            Capabilities.current_transform,
            CompositeAlpha::Opaque,
            PresentMode,
            FullscreenExclusive::Default,
            true, //Clipped
            ColorSpace::SrgbNonLinear,
        )
        .expect("Failed to create swap chain");

        (SwapChain, Images)
    }

    fn CreateRenderPass(
        Device: &Arc<Device>,
        ColorFormat: Format,
    ) -> Arc<RenderPassAbstract + Send + Sync> {
        Arc::new(
            single_pass_renderpass!(Device.clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: Store,
                        format: ColorFormat,
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {}
                }
            )
            .unwrap(),
        )
    }

    fn CreateGraphicsPipeline(
        Device: &Arc<Device>,
        Extent: [u32; 2],
        RenderPass: &Arc<RenderPassAbstract + Send + Sync>,
    ) -> Arc<ConcrateGraphicsPipeline> {
        let VertexShaderModule = VertexShader::Shader::load(Device.clone())
            .expect("Failed to create vertex shader module");
        let FragShaderModule = FragmentShader::Shader::load(Device.clone())
            .expect("Failed to create fragment shader module");

        let Dimensions = [Extent[0] as f32, Extent[1] as f32];
        let Viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: Dimensions,
            depth_range: 0.0..1.0,
        };

        Arc::new(
            GraphicsPipeline::start()
                .vertex_input(BufferlessDefinition {})
                .vertex_shader(VertexShaderModule.main_entry_point(), ())
                .triangle_list() //defaultらしい
                .primitive_restart(false)
                .viewports(vec![Viewport])
                .fragment_shader(FragShaderModule.main_entry_point(), ())
                .depth_clamp(false)
                .polygon_mode_fill()
                .line_width(1.0)
                .cull_mode_back()
                .front_face_clockwise()
                .blend_pass_through()
                .render_pass(Subpass::from(RenderPass.clone(), 0).unwrap())
                .build(Device.clone())
                .unwrap(),
        )
    }

    fn CreateFramebuffers(
        SwapchainImages: &[Arc<SwapchainImage<Window>>],
        RenderPass: &Arc<RenderPassAbstract + Send + Sync>,
    ) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
        SwapchainImages
            .iter()
            .map(|Image| {
                let FBA: Arc<FramebufferAbstract + Send + Sync> = Arc::new(
                    Framebuffer::start(RenderPass.clone())
                        .add(Image.clone())
                        .unwrap()
                        .build()
                        .unwrap(),
                );
                FBA
            })
            .collect()
    }

    fn CreateCommandBuffers(
        Device: &Arc<Device>,
        GraphicsQueue: &Arc<Queue>,
        GraphicsPipeline: &Arc<ConcrateGraphicsPipeline>,
        SwapchainFramebuffers: &Vec<Arc<FramebufferAbstract + Send + Sync>>,
    ) -> Vec<Arc<AutoCommandBuffer>> {
        let QueueFamily = GraphicsQueue.family();
        SwapchainFramebuffers
            .iter()
            .map(|FrameBuffer| {
                let Vertices = BufferlessVertices {
                    vertices: 3,
                    instances: 1,
                };
                Arc::new(
                    AutoCommandBufferBuilder::primary_simultaneous_use(Device.clone(), QueueFamily)
                        .unwrap()
                        .begin_render_pass(
                            FrameBuffer.clone(),
                            false,
                            vec![[0.0, 0.0, 0.0, 1.0].into()],
                        )
                        .unwrap()
                        .draw(
                            GraphicsPipeline.clone(),
                            &DynamicState::none(),
                            Vertices,
                            (),
                            (),
                        )
                        .unwrap()
                        .end_render_pass()
                        .unwrap()
                        .build()
                        .unwrap(),
                )
            })
            .collect()
    }

    fn CreateSyncObjects(Device: &Arc<Device>) -> Box<GpuFuture> {
        Box::new(sync::now(Device.clone())) as Box<GpuFuture>
    }

    fn Run() {
        let Instance = Self::CreateVulkanInstance();
        let DebugCallback = Self::SetupDebugCallback(&Instance);
        let EventLoop = EventLoop::new();
        let Surface = WindowBuilder::new()
            .with_title(Title)
            .with_inner_size(LogicalSize::new(Width, Height))
            .build_vk_surface(&EventLoop, Instance.clone())
            .expect("Unable to create Window.");
        let PhysicalDeviceIndex = Self::PickPhysicalDevice(&Instance, &Surface);
        let (Device, GraphicsQueue, PresentQueue) =
            Self::CreateLogicalDevice(&Instance, &Surface, PhysicalDeviceIndex);
        let (mut Swapchain, mut SwapchainImages) = Self::CreateSwapChain(
            &Instance,
            &Surface,
            PhysicalDeviceIndex,
            &Device,
            &GraphicsQueue,
            &PresentQueue,
        );
        let RenderPass = Self::CreateRenderPass(&Device, Swapchain.format());
        let GraphicsPipeline =
            Self::CreateGraphicsPipeline(&Device, Swapchain.dimensions(), &RenderPass);
        let SwapchainFramebuffers = Self::CreateFramebuffers(&SwapchainImages, &RenderPass);

        let mut PreviousFrameEnd = Some(Self::CreateSyncObjects(&Device));
        let mut RecreateSwapchain = false;

        let CommandBuffer = Self::CreateCommandBuffers(
            &Device,
            &GraphicsQueue,
            &GraphicsPipeline,
            &SwapchainFramebuffers,
        );

        EventLoop.run(move |Event, _, ControlFlow| {
            //Render
            {
                PreviousFrameEnd.as_mut().unwrap().cleanup_finished();

                if RecreateSwapchain {
                    let (NewSwapchain, NewSwapchainImages) = Swapchain
                        .recreate_with_dimensions([Width as u32, Height as u32])
                        .unwrap();
                    Swapchain = NewSwapchain;
                    SwapchainImages = NewSwapchainImages;
                }

                let (ImageIndex, _, AcquireFuture) =
                    match acquire_next_image(Swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            RecreateSwapchain = true;
                            return;
                        }
                        Err(err) => panic!("{:?}", err),
                    };
                let CommandBuffer = CommandBuffer[ImageIndex].clone();

                let Future = PreviousFrameEnd
                    .take()
                    .unwrap()
                    .join(AcquireFuture)
                    .then_execute(GraphicsQueue.clone(), CommandBuffer)
                    .unwrap()
                    .then_swapchain_present(PresentQueue.clone(), Swapchain.clone(), ImageIndex)
                    .then_signal_fence_and_flush();

                match Future {
                    Ok(Future) => {
                        PreviousFrameEnd = Some(Box::new(Future) as Box<_>);
                    }
                    Err(FlushError::OutOfDate) => {
                        RecreateSwapchain = true;
                        PreviousFrameEnd = Some(Box::new(sync::now(Device.clone())) as Box<_>);
                    }
                    Err(err) => {
                        println!("{:?}", err);
                        PreviousFrameEnd = Some(Box::new(sync::now(Device.clone())) as Box<_>);
                    }
                }
            }

            match Event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    *ControlFlow = ControlFlow::Exit;
                }
                _ => {}
            }
        });
    }

    fn DrawFrame(&mut self) {}
}

fn main() {
    App::Run();

    println!("WONT REACH HERE.")
}
