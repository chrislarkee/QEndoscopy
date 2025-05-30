# -*- coding: utf-8 -*-

###########################################################################
## Python code generated with wxFormBuilder (version 4.2.1-0-g80c4cb6)
## http://www.wxformbuilder.org/
##
## PLEASE DO *NOT* EDIT THIS FILE!
###########################################################################

import wx
#import wx.xrc
import wx.grid

###########################################################################
## Class MainInterface
###########################################################################

class MainInterface ( wx.Frame ):

    def __init__( self, parent ):
        wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"Quantitative Endoscopy", pos = wx.DefaultPosition, size = wx.Size( 1400,950 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )

        self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )
        self.SetForegroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOWTEXT ) )
        self.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOW ) )

        s_Main = wx.BoxSizer( wx.VERTICAL )

        self.p_StatusBar = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        self.p_StatusBar.SetBackgroundColour( wx.Colour( 56, 56, 56 ) )

        s_StatusBar = wx.BoxSizer( wx.HORIZONTAL )

        self.t_status = wx.StaticText( self.p_StatusBar, wx.ID_ANY, u"Open a video to begin.", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTER_HORIZONTAL )
        self.t_status.Wrap( -1 )

        self.t_status.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_ITALIC, wx.FONTWEIGHT_NORMAL, False, wx.EmptyString ) )
        self.t_status.SetForegroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_BTNHIGHLIGHT ) )

        s_StatusBar.Add( self.t_status, 1, wx.ALL|wx.EXPAND, 5 )


        self.p_StatusBar.SetSizer( s_StatusBar )
        self.p_StatusBar.Layout()
        s_StatusBar.Fit( self.p_StatusBar )
        s_Main.Add( self.p_StatusBar, 0, wx.EXPAND, 5 )

        self.p_TopArea = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        self.p_TopArea.SetForegroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOW ) )
        self.p_TopArea.SetBackgroundColour( wx.Colour( 56, 56, 56 ) )

        s_TopArea = wx.BoxSizer( wx.VERTICAL )

        s_Images = wx.BoxSizer( wx.HORIZONTAL )

        s_Images.SetMinSize( wx.Size( 220,100 ) )

        s_Images.Add( ( 0, 0), 1, wx.EXPAND, 5 )

        self.i_Image = wx.StaticBitmap( self.p_TopArea, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.Size( 300,300 ), wx.FULL_REPAINT_ON_RESIZE )
        self.i_Image.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_BACKGROUND ) )

        s_Images.Add( self.i_Image, 0, wx.EXPAND|wx.SHAPED, 2 )


        s_Images.Add( ( 0, 0), 1, wx.EXPAND, 5 )

        self.i_Depth = wx.StaticBitmap( self.p_TopArea, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.Size( 300,300 ), wx.FULL_REPAINT_ON_RESIZE )
        self.i_Depth.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_BACKGROUND ) )

        s_Images.Add( self.i_Depth, 0, wx.EXPAND|wx.SHAPED, 2 )


        s_Images.Add( ( 0, 0), 1, wx.EXPAND, 5 )


        s_TopArea.Add( s_Images, 1, wx.EXPAND|wx.TOP, 5 )

        s_transport = wx.BoxSizer( wx.HORIZONTAL )

        self.b_framePrevious = wx.Button( self.p_TopArea, wx.ID_ANY, u"<", wx.DefaultPosition, wx.Size( 60,-1 ), wx.BU_NOTEXT )

        self.b_framePrevious.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_GO_BACK, wx.ART_HELP_BROWSER ) )
        s_transport.Add( self.b_framePrevious, 0, wx.ALL, 5 )

        self.m_Time = wx.Slider( self.p_TopArea, wx.ID_ANY, 0, 0, 1, wx.DefaultPosition, wx.DefaultSize, wx.SL_HORIZONTAL|wx.SL_VALUE_LABEL )
        s_transport.Add( self.m_Time, 1, wx.EXPAND, 30 )

        self.b_FrameNext = wx.Button( self.p_TopArea, wx.ID_ANY, u">", wx.DefaultPosition, wx.Size( 60,-1 ), wx.BU_NOTEXT )

        self.b_FrameNext.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_GO_FORWARD, wx.ART_HELP_BROWSER ) )
        s_transport.Add( self.b_FrameNext, 0, wx.ALL, 5 )


        s_TopArea.Add( s_transport, 0, wx.EXPAND, 5 )


        self.p_TopArea.SetSizer( s_TopArea )
        self.p_TopArea.Layout()
        s_TopArea.Fit( self.p_TopArea )
        s_Main.Add( self.p_TopArea, 5, wx.EXPAND, 5 )

        s_BottomArea = wx.BoxSizer( wx.HORIZONTAL )


        s_BottomArea.Add( ( 0, 0), 1, 0, 5 )

        s_ToolsArea1 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Image Tools" ), wx.HORIZONTAL )

        s_grid = wx.GridSizer( 0, 1, 0, 0 )

        self.b_open = wx.Button( s_ToolsArea1.GetStaticBox(), wx.ID_ANY, u"Open Image", wx.DefaultPosition, wx.DefaultSize, 0 )

        self.b_open.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_FILE_OPEN, wx.ART_HELP_BROWSER ) )
        s_grid.Add( self.b_open, 0, wx.ALL|wx.EXPAND, 5 )

        self.b_playVideo = wx.ToggleButton( s_ToolsArea1.GetStaticBox(), wx.ID_ANY, u"Play/Pause Video", wx.DefaultPosition, wx.DefaultSize, 0 )

        self.b_playVideo.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_GO_FORWARD, wx.ART_MENU ) )
        self.b_playVideo.SetBitmapPressed( wx.ArtProvider.GetBitmap( wx.ART_DELETE, wx.ART_MENU ) )
        self.b_playVideo.Enable( False )

        s_grid.Add( self.b_playVideo, 0, wx.ALL|wx.EXPAND, 5 )

        self.b_showSettings = wx.Button( s_ToolsArea1.GetStaticBox(), wx.ID_ANY, u"Video Settings", wx.DefaultPosition, wx.DefaultSize, 0 )

        self.b_showSettings.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_FIND_AND_REPLACE, wx.ART_HELP_BROWSER ) )
        self.b_showSettings.Enable( False )

        s_grid.Add( self.b_showSettings, 0, wx.ALL|wx.EXPAND, 5 )

        self.b_exportImage = wx.Button( s_ToolsArea1.GetStaticBox(), wx.ID_ANY, u"Export Images", wx.DefaultPosition, wx.DefaultSize, 0 )

        self.b_exportImage.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_FILE_SAVE, wx.ART_HELP_BROWSER ) )
        self.b_exportImage.Enable( False )

        s_grid.Add( self.b_exportImage, 0, wx.ALL|wx.EXPAND, 5 )


        s_ToolsArea1.Add( s_grid, 0, wx.EXPAND, 5 )


        s_BottomArea.Add( s_ToolsArea1, 1, wx.ALL|wx.EXPAND, 5 )

        s_ToolsArea2 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Measurement Tools" ), wx.VERTICAL )

        gSizer4 = wx.GridSizer( 0, 2, 0, 0 )

        self.m_staticText8 = wx.StaticText( s_ToolsArea2.GetStaticBox(), wx.ID_ANY, u"Current Tool: ", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT )
        self.m_staticText8.Wrap( -1 )

        gSizer4.Add( self.m_staticText8, 0, wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, 5 )

        b_mtoolChoices = [ u"Cross Section", u"Draw Line", u"Draw Polygon" ]
        self.b_mtool = wx.Choice( s_ToolsArea2.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, b_mtoolChoices, 0 )
        self.b_mtool.SetSelection( 0 )
        gSizer4.Add( self.b_mtool, 0, wx.ALL|wx.EXPAND, 5 )


        s_ToolsArea2.Add( gSizer4, 0, wx.EXPAND, 5 )

        self.m_thresholdSlider = wx.Slider( s_ToolsArea2.GetStaticBox(), wx.ID_ANY, 1000, 0, 5000, wx.DefaultPosition, wx.DefaultSize, wx.SL_MIN_MAX_LABELS )
        self.m_thresholdSlider.SetFont( wx.Font( 8, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False, wx.EmptyString ) )

        s_ToolsArea2.Add( self.m_thresholdSlider, 0, wx.ALL|wx.EXPAND, 5 )

        gSizer5 = wx.GridSizer( 0, 2, 0, 0 )

        self.b_clearMeasurements = wx.Button( s_ToolsArea2.GetStaticBox(), wx.ID_ANY, u"Clear Overlay", wx.DefaultPosition, wx.DefaultSize, 0 )

        self.b_clearMeasurements.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_MISSING_IMAGE, wx.ART_HELP_BROWSER ) )
        self.b_clearMeasurements.Enable( False )

        gSizer5.Add( self.b_clearMeasurements, 0, wx.ALL|wx.EXPAND, 5 )

        self.b_recordMeasurement = wx.Button( s_ToolsArea2.GetStaticBox(), wx.ID_ANY, u"Record Measurement", wx.DefaultPosition, wx.DefaultSize, 0 )

        self.b_recordMeasurement.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_PLUS, wx.ART_HELP_BROWSER ) )
        self.b_recordMeasurement.Enable( False )

        gSizer5.Add( self.b_recordMeasurement, 0, wx.ALL|wx.EXPAND, 5 )

        self.b_showExporter = wx.Button( s_ToolsArea2.GetStaticBox(), wx.ID_ANY, u"Review Measurements", wx.DefaultPosition, wx.DefaultSize, 0 )

        self.b_showExporter.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_FILE_SAVE_AS, wx.ART_HELP_BROWSER ) )
        self.b_showExporter.Enable( False )

        gSizer5.Add( self.b_showExporter, 0, wx.ALL|wx.EXPAND, 5 )

        self.b_help = wx.Button( s_ToolsArea2.GetStaticBox(), wx.ID_ANY, u"Help", wx.DefaultPosition, wx.DefaultSize, 0 )

        self.b_help.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_HELP, wx.ART_MENU ) )
        gSizer5.Add( self.b_help, 0, wx.ALL|wx.EXPAND, 5 )


        s_ToolsArea2.Add( gSizer5, 0, wx.EXPAND, 5 )


        s_BottomArea.Add( s_ToolsArea2, 2, wx.ALL|wx.EXPAND, 5 )

        s_MeasurementsArea = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Measurement Data" ), wx.VERTICAL )

        self.t_statusText = wx.TextCtrl( s_MeasurementsArea.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_MULTILINE|wx.TE_READONLY|wx.TE_WORDWRAP|wx.BORDER_NONE )
        s_MeasurementsArea.Add( self.t_statusText, 1, wx.EXPAND, 5 )


        s_BottomArea.Add( s_MeasurementsArea, 3, wx.ALL|wx.EXPAND, 5 )


        s_BottomArea.Add( ( 0, 0), 1, wx.EXPAND, 5 )


        s_Main.Add( s_BottomArea, 0, wx.EXPAND, 5 )


        self.SetSizer( s_Main )
        self.Layout()

        self.Centre( wx.BOTH )

        # Connect Events
        self.Bind( wx.EVT_CLOSE, self.saveAndQuit )
        self.Bind( wx.EVT_MAXIMIZE, self.setSize )
        self.Bind( wx.EVT_MOVE_END, self.setSize )
        self.Bind( wx.EVT_SHOW, self.setSize )
        self.i_Image.Bind( wx.EVT_LEFT_DOWN, self.pickPoint )
        self.i_Image.Bind( wx.EVT_RIGHT_DOWN, self.clearMeasurements )
        self.i_Depth.Bind( wx.EVT_LEFT_DOWN, self.pickPoint )
        self.i_Depth.Bind( wx.EVT_RIGHT_DOWN, self.clearMeasurements )
        self.b_framePrevious.Bind( wx.EVT_BUTTON, self.framePrevious )
        self.m_Time.Bind( wx.EVT_SCROLL_CHANGED, self.scrubDone )
        self.m_Time.Bind( wx.EVT_SLIDER, self.scrub )
        self.b_FrameNext.Bind( wx.EVT_BUTTON, self.frameNext )
        self.b_open.Bind( wx.EVT_BUTTON, self.openVideo )
        self.b_playVideo.Bind( wx.EVT_TOGGLEBUTTON, self.playVideo )
        self.b_showSettings.Bind( wx.EVT_BUTTON, self.showSettings )
        self.b_exportImage.Bind( wx.EVT_BUTTON, self.exportImage )
        self.b_mtool.Bind( wx.EVT_CHOICE, self.changeTool )
        self.m_thresholdSlider.Bind( wx.EVT_SCROLL, self.threshChange )
        self.b_clearMeasurements.Bind( wx.EVT_BUTTON, self.clearMeasurements )
        self.b_recordMeasurement.Bind( wx.EVT_BUTTON, self.recordMeasurement )
        self.b_showExporter.Bind( wx.EVT_BUTTON, self.showExporter )
        self.b_help.Bind( wx.EVT_BUTTON, self.openHelp )
        self.t_statusText.Bind( wx.EVT_LEFT_DOWN, self.copyText )

    def __del__( self ):
        pass


    # Virtual event handlers, override them in your derived class
    def saveAndQuit( self, event ):
        pass

    def setSize( self, event ):
        pass



    def pickPoint( self, event ):
        pass

    def clearMeasurements( self, event ):
        pass



    def framePrevious( self, event ):
        pass

    def scrubDone( self, event ):
        pass

    def scrub( self, event ):
        pass

    def frameNext( self, event ):
        pass

    def openVideo( self, event ):
        pass

    def playVideo( self, event ):
        pass

    def showSettings( self, event ):
        pass

    def exportImage( self, event ):
        pass

    def changeTool( self, event ):
        pass

    def threshChange( self, event ):
        pass


    def recordMeasurement( self, event ):
        pass

    def showExporter( self, event ):
        pass

    def openHelp( self, event ):
        pass

    def copyText( self, event ):
        pass


###########################################################################
## Class VideoSettings
###########################################################################

class VideoSettings ( wx.Dialog ):

    def __init__( self, parent ):
        wx.Dialog.__init__ ( self, parent, id = wx.ID_ANY, title = u"Video Settings Editor", pos = wx.DefaultPosition, size = wx.Size( 1200,750 ), style = wx.DEFAULT_DIALOG_STYLE|wx.RESIZE_BORDER )

        self.SetSizeHints( wx.Size( -1,-1 ), wx.DefaultSize )

        s_Trimmer = wx.BoxSizer( wx.VERTICAL )

        fgSizer1 = wx.FlexGridSizer( 3, 2, 0, 0 )
        fgSizer1.AddGrowableCol( 0 )
        fgSizer1.AddGrowableCol( 1 )
        fgSizer1.AddGrowableRow( 1 )
        fgSizer1.SetFlexibleDirection( wx.BOTH )
        fgSizer1.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_ALL )

        self.t_Start = wx.StaticText( self, wx.ID_ANY, u"Start Frame", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTER_HORIZONTAL )
        self.t_Start.Wrap( -1 )

        fgSizer1.Add( self.t_Start, 1, wx.ALIGN_CENTER, 5 )

        self.t_End = wx.StaticText( self, wx.ID_ANY, u"End Frame", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTER_HORIZONTAL )
        self.t_End.Wrap( -1 )

        fgSizer1.Add( self.t_End, 1, wx.ALIGN_CENTER, 5 )

        self.i_TrimStart = wx.StaticBitmap( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.Size( 500,500 ), 0 )
        self.i_TrimStart.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_BTNTEXT ) )

        fgSizer1.Add( self.i_TrimStart, 1, wx.ALIGN_CENTER|wx.SHAPED, 5 )

        self.i_TrimEnd = wx.StaticBitmap( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.Size( 500,500 ), 0 )
        self.i_TrimEnd.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_BTNTEXT ) )

        fgSizer1.Add( self.i_TrimEnd, 1, wx.ALIGN_CENTER|wx.SHAPED, 5 )

        self.b_TrimStart = wx.Slider( self, wx.ID_ANY, 0, 0, 100, wx.DefaultPosition, wx.DefaultSize, wx.SL_HORIZONTAL|wx.SL_VALUE_LABEL )
        fgSizer1.Add( self.b_TrimStart, 1, wx.EXPAND, 5 )

        self.b_TrimEnd = wx.Slider( self, wx.ID_ANY, 100, 0, 100, wx.DefaultPosition, wx.DefaultSize, wx.SL_HORIZONTAL|wx.SL_VALUE_LABEL )
        fgSizer1.Add( self.b_TrimEnd, 1, wx.EXPAND, 5 )


        s_Trimmer.Add( fgSizer1, 0, wx.ALL|wx.EXPAND, 5 )

        self.t_trimTitleLine2 = wx.StaticLine( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
        s_Trimmer.Add( self.t_trimTitleLine2, 0, wx.EXPAND |wx.ALL, 5 )

        bSizer14 = wx.BoxSizer( wx.HORIZONTAL )


        bSizer14.Add( ( 0, 0), 1, wx.EXPAND, 5 )

        self.t_crop = wx.StaticText( self, wx.ID_ANY, u"Horizontal Offset:", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT )
        self.t_crop.Wrap( -1 )

        bSizer14.Add( self.t_crop, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5 )

        self.b_cropOffset = wx.SpinCtrl( self, wx.ID_ANY, u"0", wx.Point( -1,-1 ), wx.Size( 150,-1 ), wx.ALIGN_CENTER_HORIZONTAL|wx.SP_ARROW_KEYS, -300, 300, 0 )
        bSizer14.Add( self.b_cropOffset, 0, wx.ALIGN_CENTER_VERTICAL, 5 )


        bSizer14.Add( ( 0, 0), 1, wx.EXPAND, 5 )

        self.t_zoom = wx.StaticText( self, wx.ID_ANY, u"Zoom:", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT )
        self.t_zoom.Wrap( -1 )

        bSizer14.Add( self.t_zoom, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5 )

        self.b_zoom = wx.SpinCtrl( self, wx.ID_ANY, u"0", wx.Point( -1,-1 ), wx.Size( 150,-1 ), wx.ALIGN_CENTER_HORIZONTAL|wx.SP_ARROW_KEYS, 0, 250, 0 )
        bSizer14.Add( self.b_zoom, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5 )


        bSizer14.Add( ( 0, 0), 1, wx.EXPAND, 5 )

        self.t_speed = wx.StaticText( self, wx.ID_ANY, u"Playback Speed:", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT )
        self.t_speed.Wrap( -1 )

        bSizer14.Add( self.t_speed, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5 )

        self.b_speed = wx.SpinCtrlDouble( self, wx.ID_ANY, u"1", wx.DefaultPosition, wx.Size( 150,-1 ), wx.ALIGN_CENTER_HORIZONTAL|wx.SP_ARROW_KEYS, 0.1, 4, 1, 0.2 )
        self.b_speed.SetDigits( 2 )
        bSizer14.Add( self.b_speed, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5 )


        bSizer14.Add( ( 0, 0), 1, wx.EXPAND, 5 )

        self.b_colormap = wx.CheckBox( self, wx.ID_ANY, u"Use Colormap", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT )
        self.b_colormap.SetValue(True)
        bSizer14.Add( self.b_colormap, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5 )


        bSizer14.Add( ( 0, 0), 1, wx.EXPAND, 5 )


        s_Trimmer.Add( bSizer14, 1, wx.ALL|wx.EXPAND, 5 )

        self.t_trimTitleLine1 = wx.StaticLine( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
        s_Trimmer.Add( self.t_trimTitleLine1, 0, wx.EXPAND |wx.ALL, 5 )

        bSizer10 = wx.BoxSizer( wx.HORIZONTAL )

        self.t_duration = wx.StaticText( self, wx.ID_ANY, u"Clip Duration:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.t_duration.Wrap( -1 )

        bSizer10.Add( self.t_duration, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5 )


        bSizer10.Add( ( 0, 0), 1, wx.EXPAND, 5 )

        self.b_TrimDone = wx.Button( self, wx.ID_ANY, u"Apply Changes", wx.DefaultPosition, wx.DefaultSize, 0 )

        self.b_TrimDone.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_FILE_SAVE, wx.ART_HELP_BROWSER ) )
        self.b_TrimDone.SetMinSize( wx.Size( 200,-1 ) )

        bSizer10.Add( self.b_TrimDone, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5 )


        s_Trimmer.Add( bSizer10, 1, wx.ALL|wx.EXPAND, 5 )


        self.SetSizer( s_Trimmer )
        self.Layout()

        self.Centre( wx.BOTH )

        # Connect Events
        self.Bind( wx.EVT_CLOSE, self.doneTrimming )
        self.b_TrimStart.Bind( wx.EVT_SLIDER, self.updateTrim )
        self.b_TrimEnd.Bind( wx.EVT_SLIDER, self.updateTrim )
        self.b_cropOffset.Bind( wx.EVT_SPINCTRL, self.updateCrop )
        self.b_cropOffset.Bind( wx.EVT_TEXT, self.updateCrop )
        self.b_cropOffset.Bind( wx.EVT_TEXT_ENTER, self.updateCrop )
        self.b_zoom.Bind( wx.EVT_SPINCTRL, self.updateCrop )
        self.b_zoom.Bind( wx.EVT_TEXT, self.updateCrop )
        self.b_zoom.Bind( wx.EVT_TEXT_ENTER, self.updateCrop )
        self.b_TrimDone.Bind( wx.EVT_BUTTON, self.doneTrimming )

    def __del__( self ):
        pass


    # Virtual event handlers, override them in your derived class
    def doneTrimming( self, event ):
        pass

    def updateTrim( self, event ):
        pass


    def updateCrop( self, event ):
        pass








###########################################################################
## Class Exporter
###########################################################################

class Exporter ( wx.Dialog ):

    def __init__( self, parent ):
        wx.Dialog.__init__ ( self, parent, id = wx.ID_ANY, title = u"Export Preview", pos = wx.DefaultPosition, size = wx.Size( 1100,800 ), style = wx.DEFAULT_DIALOG_STYLE|wx.RESIZE_BORDER )

        self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )

        s_exportMain = wx.BoxSizer( wx.VERTICAL )

        self.grid = wx.grid.Grid( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, 0 )

        # Grid
        self.grid.CreateGrid( 1, 9 )
        self.grid.EnableEditing( False )
        self.grid.EnableGridLines( True )
        self.grid.EnableDragGridSize( False )
        self.grid.SetMargins( 20, 20 )

        # Columns
        self.grid.SetColSize( 0, 50 )
        self.grid.SetColSize( 1, 70 )
        self.grid.SetColSize( 2, 140 )
        self.grid.SetColSize( 3, 140 )
        self.grid.SetColSize( 4, 100 )
        self.grid.SetColSize( 5, 100 )
        self.grid.SetColSize( 6, 110 )
        self.grid.SetColSize( 7, 110 )
        self.grid.SetColSize( 8, 200 )
        self.grid.EnableDragColMove( False )
        self.grid.EnableDragColSize( True )
        self.grid.SetColLabelValue( 0, u"i" )
        self.grid.SetColLabelValue( 1, u"Frame" )
        self.grid.SetColLabelValue( 2, u"Timecode" )
        self.grid.SetColLabelValue( 3, u"MinMax" )
        self.grid.SetColLabelValue( 4, u"Threshold" )
        self.grid.SetColLabelValue( 5, u"Distance" )
        self.grid.SetColLabelValue( 6, u"Area (px)" )
        self.grid.SetColLabelValue( 7, u"Area (cm^2)" )
        self.grid.SetColLabelValue( 8, u"CameraPos" )
        self.grid.SetColLabelSize( wx.grid.GRID_AUTOSIZE )
        self.grid.SetColLabelAlignment( wx.ALIGN_CENTER, wx.ALIGN_CENTER )

        # Rows
        self.grid.SetRowSize( 0, 0 )
        self.grid.EnableDragRowSize( False )
        self.grid.SetRowLabelSize( 0 )
        self.grid.SetRowLabelAlignment( wx.ALIGN_CENTER, wx.ALIGN_CENTER )

        # Label Appearance

        # Cell Defaults
        self.grid.SetDefaultCellFont( wx.Font( 9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False, wx.EmptyString ) )
        self.grid.SetDefaultCellAlignment( wx.ALIGN_LEFT, wx.ALIGN_CENTER )
        s_exportMain.Add( self.grid, 1, wx.ALL|wx.EXPAND, 10 )

        self.m_staticline3 = wx.StaticLine( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
        s_exportMain.Add( self.m_staticline3, 0, wx.ALL|wx.EXPAND, 5 )

        s_exportBottom = wx.BoxSizer( wx.HORIZONTAL )

        s_exportBottom.SetMinSize( wx.Size( -1,50 ) )
        self.b_exportFilename = wx.FilePickerCtrl( self, wx.ID_ANY, wx.EmptyString, u"Select a file to save an XLSX file.", u"*.xlsx", wx.DefaultPosition, wx.DefaultSize, wx.FLP_OVERWRITE_PROMPT|wx.FLP_SAVE|wx.FLP_USE_TEXTCTRL )
        s_exportBottom.Add( self.b_exportFilename, 1, wx.EXPAND|wx.RIGHT, 10 )

        self.b_exportSave = wx.Button( self, wx.ID_ANY, u"Save", wx.DefaultPosition, wx.DefaultSize, 0 )

        self.b_exportSave.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_FILE_SAVE, wx.ART_HELP_BROWSER ) )
        self.b_exportSave.Enable( False )
        self.b_exportSave.SetMinSize( wx.Size( 200,-1 ) )

        s_exportBottom.Add( self.b_exportSave, 0, wx.EXPAND, 5 )


        s_exportMain.Add( s_exportBottom, 0, wx.ALL|wx.EXPAND, 5 )


        self.SetSizer( s_exportMain )
        self.Layout()

        self.Centre( wx.BOTH )

        # Connect Events
        self.b_exportFilename.Bind( wx.EVT_FILEPICKER_CHANGED, self.readyToSave )
        self.b_exportSave.Bind( wx.EVT_BUTTON, self.exportSave )

    def __del__( self ):
        pass


    # Virtual event handlers, override them in your derived class
    def readyToSave( self, event ):
        pass

    def exportSave( self, event ):
        pass


